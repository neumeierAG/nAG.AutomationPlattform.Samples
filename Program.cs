using Backend.Scripting;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion.Internal;
using Microsoft.Extensions.AI;
using Microsoft.OData.Edm;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using nAG.Framework.SapBusinessOne.Models;
using s2industries.ZUGFeRD;
using System.Runtime.CompilerServices;
using System;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using static Backend.Scripting.AutomationUtils;
using AiImportVendor;
using Npgsql.Internal;
using System.Net.Mail;
using Microsoft.OData.UriParser;

try
{
    //logic start logs
    DateTime startTime = DateTime.Now;
    await WriteLog(new()
    {
        Status = Backend.Database.RequestStatus.Success,
        StatusMessage = "Skript wird ausgeführt.",
        TransactionStart = startTime
    });

    DateTime filecheckStarttime = DateTime.Now;
    //get file from filedrop
    IFormFile file = AutomationWebhookContext?.Request.Form.Files[0] ?? throw new Exception("No file uploaded");

    using var stream = file.OpenReadStream();
    using var invoiceStream = new MemoryStream();
    await stream.CopyToAsync(invoiceStream);
    invoiceStream.Position = 0;

    InvoiceDescriptor invoice = InvoiceDescriptor.Load(invoiceStream);

    //file type check
    string formatLabel = invoice.Profile == Profile.XRechnung ? "XRechnung" : $"ZUGFeRD {invoice.Profile}";

    /*if(invoice.Profile == Profile.XRechnung)
    {
        formatLabel = "XRechnung";
    }
    else if(invoice.Profile == Profile.ZUGFeRD)
    {
        formatLabel = $"ZUGFeRD {invoice.Profile}";
    }
    else
    {
        throw new Exception();
    }*/

    //log entry
    await WriteLog(new()
    {
        Status = Backend.Database.RequestStatus.Success,
        StatusMessage = $"{formatLabel}{(string.IsNullOrWhiteSpace(invoice.Name) ? "" : $" '{invoice.Name}'")} erkannt.",
        TransactionStart = filecheckStarttime
    });

    //azure connection credentials
    string azureEndpoint = "https://xxxxx.openai.azure.com/";
    string azureApiKey = "xxxxxx";

    //disable warning for "subject to change or removal in the future" warning
#pragma warning disable SKEXP0010
    //central object in semantic kernel (open-source-sdk from microsoft for AI-integration)
    var builder = Kernel.CreateBuilder();

    //embedding generator for transforming text into numeric vectors
    builder.AddAzureOpenAIEmbeddingGenerator(
        serviceId: "text-embedding-3-large",
        deploymentName: "text-embedding-3-large",
        endpoint: azureEndpoint,
        apiKey: azureApiKey
    );

    //chat-completion model for chat-reply generation
    builder.AddAzureOpenAIChatCompletion(
        serviceId: "gpt-4.1",
        deploymentName: "gpt-4.1",
        endpoint: azureEndpoint,
        apiKey: azureApiKey
    );

    //enable warning again
#pragma warning restore SKEXP0010
    // using build kernel
    var kernel = builder.Build();
    //embedding service for context/similarity comparison
    var embeddingService = kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
    var chatService = kernel.GetRequiredService<IChatCompletionService>();

    //get SAP-Client
    AutomationSapClient sapClient = GetSapClient();

    //Business Partners
    DateTime vendorVectorStarttime = DateTime.Now;
    //only gets suppliers that start with card name c
    List<VendorVectorItem> vendorVectorItems = await sapClient.Get<VendorVectorItem>(
        "BusinessPartners?$filter=CardType eq 'S' and contains(CardName, 'lie')" +
        "&$select=CardCode,CardName,EmailAddress,ContactPerson,Address,ZipCode,City,Country,HouseBankIBAN,FederalTaxID,VatIDNum"
    );

    //create BP embeddings and saves it in dictionary
    var vendorEmbeddings = new Dictionary<string, Embedding<float>>();
    foreach (var vendorVectorItem in vendorVectorItems)
    {
        // create readable text for AI
        string semanticText = vendorVectorItem.ToSemanticString();
        if (string.IsNullOrWhiteSpace(semanticText))
            continue;

        //depending on the semantic-kernel-version there may exist an overload with or without the "kernel" parameter
        //both variants are common; if one does not compile -> try the other one
        // sends semanticText to embeddingSerivce for vector number calculation
        Embedding<float> embedding = await embeddingService.GenerateAsync(semanticText);
        //log entry
        /*await WriteLog(new()
        {
            Status = Backend.Database.RequestStatus.Success,
            StatusMessage = $"KI-BP: {semanticText}"
        });*/
        //saves embedding to the corresponding card code
        vendorEmbeddings[vendorVectorItem.CardCode ?? Guid.NewGuid().ToString()] = embedding;
    }

    //log entry
    await WriteLog(new()
    {
        Status = Backend.Database.RequestStatus.Success,
        StatusMessage = $"KI-BP: {vendorEmbeddings.Count} Vektoren erstellt.", //für Anbieter: {string.Join(", ", vendorEmbeddings.Keys)}
        TransactionStart = vendorVectorStarttime
    });

    //Items
    DateTime itemVectorStarttime = DateTime.Now;
    //only get valid purchase items with cks in their names
    List<ItemVectorItem> itemVectorItems = await sapClient.Get<ItemVectorItem>("Items?$filter=contains(ItemName,'Systemhaus')&$expand=ItemGroups&$select=ItemGroups/GroupName"); //PurchaseItem eq 'tYES' and Valid eq 'tYES' and contains(ItemName,'cks')&$expand=ItemGroups&$select=ItemGroups/GroupName
    //create item embeddings and save them in dictionary
    var itemEmbeddings = new Dictionary<string, Embedding<float>>();
    foreach (var itemVectorItem in itemVectorItems)
    {
        if (string.IsNullOrWhiteSpace(itemVectorItem.ItemCode))
            continue;
        string semanticText = itemVectorItem.ToSemanticString();
        if (string.IsNullOrWhiteSpace(semanticText))
            continue;

        //depending on the semantic-kernel-version there may exist an overload with or without the "kernel" parameter
        //both variants are common; if one does not compile -> try the other one
        // sends semanticText to embeddingSerivce for vector number calculation
        Embedding<float> embedding = await embeddingService.GenerateAsync(semanticText);
        //log entry
        /*await WriteLog(new()
        {
            Status = Backend.Database.RequestStatus.Success,
            StatusMessage = $"KI-ITM: {semanticText}"
        });*/
        //saves embedding to the corresponding item code
        itemEmbeddings[itemVectorItem.ItemCode] = embedding;
    }

    //log entry
    await WriteLog(new()
    {
        Status = Backend.Database.RequestStatus.Success,
        StatusMessage = $"KI-ITM: {itemEmbeddings.Count} Vektoren erstellt.",
        TransactionStart = itemVectorStarttime
    });

    //SupplierCatalogNr Match
    DateTime supplierCatalogNrMatchStartTime = DateTime.Now;
    string resolveCardCode = null;
    var invoiceSupplierIds = BusinessPartnerAiHelper.ExtractSupplierIds(invoice);

    await WriteLog(new()
    {
        Status = Backend.Database.RequestStatus.Success,
        StatusMessage = $"Überprüfe folgende BP-ID(s) auf Übereinstimmung: {string.Join(", ", invoiceSupplierIds)}", //Kommentar anpassen
        TransactionStart = supplierCatalogNrMatchStartTime
    });

    var supplierMatch = vendorVectorItems
        .Where(v => invoiceSupplierIds.Contains((v.SupplierCatalogNr ?? "").Trim().ToLowerInvariant())
                || invoiceSupplierIds.Contains((v.FederalTaxID ?? "").Trim().ToLowerInvariant())
                || invoiceSupplierIds.Contains((v.VatIDNum ?? "").Trim().ToLowerInvariant())
        )
        .ToList();

    if (supplierMatch.Count == 1)
    {
        string bp = supplierMatch[0].CardCode!;
        await WriteLog(new()
        {
            Status = Backend.Database.RequestStatus.Success,
            StatusMessage = $"BP-Katalognummer eindeutig → KI übersprungen. BP = {bp}"
        });

        resolveCardCode = bp;
    }
    else if (supplierMatch.Count > 1)
    {
        await WriteLog(new()
        {
            Status = Backend.Database.RequestStatus.Warning,
            StatusMessage = $"Mehrere BPs mit gleicher Katalognummer → KI entscheidet."
        });
    }
    else
    {
        await WriteLog(new()
        {
            Status = Backend.Database.RequestStatus.None,
            StatusMessage = $"Keine BP-Katalognummer gefunden → KI übernimmt."
        });
    }

    if (resolveCardCode == null)
    {
        //choose best fit BP 
        resolveCardCode = await BusinessPartnerAiHelper.ResolveCardCodeAsync(
            invoice,
            vendorVectorItems,
            vendorEmbeddings,
            embeddingService,
            chatService,
            fallbackCardCode: "V10000",
            logger: (msg, sts) => WriteLog(new()
            {
                Status = sts,
                StatusMessage = msg
            })
        );
    }

    DateTime sapImportStartTime = DateTime.Now;

    string documentNotes = string.Join("; ", invoice.Notes
                                            .Where(n => !string.IsNullOrWhiteSpace(n.Content))
                                            .Select(n => n.Content));

    //cut note length
    if (documentNotes.Length > 250)
    {
        documentNotes = documentNotes.Substring(0, 250);
    }

    //create purchase invoice
    Document purchaseInvoice = new()
    {
        DocObjectCode = BoObjectTypes.oPurchaseInvoices,
        CardCode = resolveCardCode,
        CardName = invoice.Seller.Name,
        Comments = documentNotes,
        NumAtCard = $"{invoice.InvoiceNo} / {invoice.OrderNo}",
        DocumentLines = new List<DocumentLine>()
    };

    //check billing Lines and add matching articles
    foreach (TradeLineItem tradeLineItem in invoice.TradeLineItems)
    {
        string matchedItemCode = await ItemAiHelper.ResolveItemCodeAsync(
            tradeLineItem,
            itemVectorItems,
            itemEmbeddings,
            embeddingService,
            chatService,
            fallbackItemCode: "E10000",
            logger: (msg, sts) => WriteLog(new()
            {
                Status = sts,
                StatusMessage = msg
            })
        );

        purchaseInvoice.DocumentLines.Add(new()
        {
            ItemCode = matchedItemCode,
            ItemDescription = $"{tradeLineItem.Name} {tradeLineItem.Description} {tradeLineItem.BuyerOrderReferencedDocument.LineID}",
            Quantity = (double?)tradeLineItem.BilledQuantity,
            UnitPrice = (double?)tradeLineItem.NetUnitPrice,
            DiscountPercent = (double?)tradeLineItem.TradeAllowanceCharges.FirstOrDefault()?.ChargePercentage,
            ItemDetails = tradeLineItem.Description
        });
    }

    //reset Reader to 0 to ensure that the entire file gets transferred
    invoiceStream.Position = 0;
    //upload E-Rechnung.xml to SAP
    Attachments2 attachment = await sapClient.PostFiles<Attachments2>("Attachments2", new() { { $"einvoice_{DateTime.Now:yyyyMMdd_HHmmss}.xml", invoiceStream } });
    //connect xml with corresponding purchase invoice
    purchaseInvoice.AttachmentEntry = attachment.AbsoluteEntry;

    //upload purchaseInvoice to SAP (geparkte Eingangsrechnung) and save response
    Document purchaseInvoiceResponse = await sapClient.Post<Document>("Drafts", purchaseInvoice);

    //final log, created SAP purchase invoice draft
    await WriteLog(new()
    {
        EntityId = purchaseInvoiceResponse.DocEntry.ToString(),
        EntityType = Backend.Database.EntityType.PurchaseInvoice,
        TransactionType = Backend.Database.TransactionType.Add,
        Status = Backend.Database.RequestStatus.Success,
        StatusMessage = $"SAP: Geprakte Eingangsrechnung {purchaseInvoiceResponse.DocNum} erstellt.",
        TransactionStart = sapImportStartTime,
        Variables = new Dictionary<string, object>
            {
                { "documentType", "Draft" }
            }
    });
}
catch
{
    //throw automatically triggers existing automation logs
    throw;
}