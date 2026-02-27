using System.Collections.Generic;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel.ChatCompletion;
using s2industries.ZUGFeRD;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace AiImportVendor
{
    public static class BusinessPartnerAiHelper
    {
        /// <summary>
        /// Gets the best matching SAP card code via AI
        /// </summary>
        public static async Task<string> ResolveCardCodeAsync(
            InvoiceDescriptor invoice,
            List<VendorVectorItem> vendors,
            Dictionary<string, Embedding<float>> vendorEmbeddings,
            IEmbeddingGenerator<string, Embedding<float>> embeddingService,
            IChatCompletionService chatService,
            string fallbackCardCode = "V10000",
            Func<string, Backend.Database.RequestStatus, Task>? logger = null)
        {        
            try
            {
                //fallbacks before calculation
                if(invoice == null)
                {
                    if(logger != null) await logger("❌BP-Match failed → invoice == null", Backend.Database.RequestStatus.Warning);
                    return fallbackCardCode;
                }
                var s = invoice.Seller;
                if(s == null)
                {
                    if(logger != null) await logger("❌BP-Match failed → invoice.Seller == null", Backend.Database.RequestStatus.Warning);
                    return fallbackCardCode;
                }
                if(vendors == null || vendors.Count == 0)
                {
                    if(logger != null) await logger("❌BP-Match failed → kein Lieferant vorhanden", Backend.Database.RequestStatus.Warning);
                    return fallbackCardCode;
                }
                if(vendorEmbeddings == null || vendorEmbeddings.Count == 0)
                {
                    if(logger != null) await logger("❌BP-Match failed → kein Ergebnis aus vendorEmbeddings", Backend.Database.RequestStatus.Warning);
                    return fallbackCardCode;
                }
                if(s == null || vendors.Count == 0 || vendorEmbeddings.Count == 0)
                {
                    return fallbackCardCode;
                }
                
                //1)
                //unique tax values
                string? sellerVat       = GetVatID(invoice);
                string? sellerTax       = GetTaxNr(invoice);
                string  sellerName      = s.Name?.Trim() ?? "";
                string? sellerCity      = s?.City?.Trim();
                string? sellerCountry   = s?.Country?.ToString();

                //2)
                //create seller embedding
                string sellerSemantic = ToSemanticStringSeller(invoice);
                await logger($"KI-BP-Info: {sellerSemantic}", Backend.Database.RequestStatus.Success);
                var queryEmbedding = await embeddingService.GenerateAsync(sellerSemantic);

                //3)
                //seller score calculation
                var scored = new List<(string CardCode, double Score, string Reason)>();

                foreach (var v in vendors)
                {
                    var code = v.CardCode ?? "";
                    if (!vendorEmbeddings.TryGetValue(code, out var emb))
                        continue;
                    
                    double cosine = CosineSimilarity(queryEmbedding.Vector.ToArray(), emb.Vector.ToArray()); //between 0 and 1
                    double boost = 0.0;
                    var reasons = new List<string> { $"cos={cosine:F3}" };

                    //VAT/UID exact match (highscore)
                    if (!string.IsNullOrWhiteSpace(sellerVat) && SameVat(sellerVat, v.VatIDNum))
                    {
                        boost += 0.50;
                        reasons.Add("VAT=match(+0.50)");
                    }

                    //Tax/Federal ID decent match (middle score)
                    if (!string.IsNullOrWhiteSpace(sellerTax) && SameId(sellerTax, v.FederalTaxID))
                    {
                        boost += 0.20;
                        reasons.Add("TaxNr=match(+0.20)");
                    }

                    //similar names (low-middle score)
                    double nameSimilar = NameSimilarityIndex(NormalizeName(sellerName), NormalizeName(v.CardName ?? ""));
                    if (nameSimilar >= 0.9)
                    {
                        boost += 0.20;
                        reasons.Add($"Name≈{nameSimilar:F2}(+0.20)");
                    }
                    else if (nameSimilar >= 0.75)
                    {
                        boost += 0.10;
                        reasons.Add($"Name≈{nameSimilar:F2}(+0.10)");
                    }
                    else if (nameSimilar >= 0.5)
                    {
                        boost += 0.05;
                        reasons.Add($"Name≈{nameSimilar:F2}(+0.05)");
                    }

                    //city/country match (low score)
                    if (!string.IsNullOrWhiteSpace(sellerCity) && SameText(sellerCity, FirstCity(v)))
                    {
                        boost += 0.05;
                        reasons.Add("City=match(+0.05)");
                    }
                    if (!string.IsNullOrWhiteSpace(sellerCountry) && SameText(sellerCountry, FirstCountry(v)))
                    {
                        boost += 0.05;
                        reasons.Add("Country=match(+0.05)");
                    }

                    //final score = 70% cosine + booster
                    double final = Math.Min(1.0, 0.70 * cosine + boost);
                    scored.Add((code, final, string.Join(", ", reasons)));
                }

                //sort based on score
                var ordered = scored.OrderByDescending(x => x.Score).ToList();
                var best = ordered.FirstOrDefault();
                var second = ordered.Skip(1).FirstOrDefault();
                
                //4)
                //threshholds; either highscore or clear difference to 2nd best
                if (best.CardCode != null)
                {
                    double delta = best.Score - (second.CardCode == null ? 0 : second.Score);

                    if (best.Score >= 0.85 || (best.Score >= 0.75 && delta >= 0.10))
                    {
                        if (logger != null) await logger($"✅BP-Match erfolgreich {best.CardCode}: score={best.Score:F2} Δ= {delta:F2} [{best.Reason}]", Backend.Database.RequestStatus.Success);
                        return best.CardCode;
                    }
                }

                //5)
                //LLM Fallback: Ask AI for top-3
                var top3 = ordered.Take(3).ToList();
                if(top3.Count > 0)
                {
                    //english for token efficiency and performance
                    string prompt = $@"
                        There is a supplier from an Invoice (semantic summary):
                        ""{sellerSemantic}""
                        
                        That are the 3 most similar Business Partners from SAP (CardCode: Name | VAT | City | Country):
                        {string.Join("\n", top3.Select((m, i) => $"{i + 1}. {m.CardCode}: {(vendors.First(v => v.CardCode == m.CardCode)).ToSemanticString()}"))}
                        
                        Only return the CardCode of the single best match.
                        If there is none, return '{fallbackCardCode}'.
                        ";
                    var llm = await chatService.GetChatMessageContentAsync(prompt);
                    var pick = llm?.Content?.Trim();

                    if (!string.IsNullOrWhiteSpace(pick))
                    {
                        if (logger != null) 
                        {
                            await logger($@"✅BP-Match (LLM) erfolgreich → Pick: {pick} 
                                (Top3: {string.Join(">> ", top3.Select((t, i) => $"<< #{i + 1}: {(vendors.First(v => v.CardCode == t.CardCode)).ToSemanticString()}"))})",
                                Backend.Database.RequestStatus.Success);
                            return pick!;
                        }
                    }
                }

                //6)
                //final fallback
                if(logger != null) await logger("❌BP-Match failed → Letzter Fallback verwendet.", Backend.Database.RequestStatus.Warning);
                return fallbackCardCode;
            }
            catch (Exception ex)
            {
                if (logger != null) await logger($"❌BP-Match Exception: {ex.GetType().Name}: {ex.Message}.", Backend.Database.RequestStatus.Warning);
                return fallbackCardCode;
            }
        }

        ///////////////////////////////////////////////////////// Help Methods //////////////////////////////////////////////////////////////
        //help method for readability (seller information | ...)
        static string ToSemanticStringSeller(InvoiceDescriptor invoice)
        {
            var parts = new List<string>();
            var s = invoice?.Seller;
            if (s == null) 
            {
                return string.Empty;
            }

            /*if (!string.IsNullOrWhiteSpace(s.ID.ToString()))
                parts.Add($"Code: {s.ID}");*/
            // Code (robust: s.ID?.ID falls vorhanden, sonst ToString)
            string? code = null;
            try
            {
                code = (s.ID is null) ? null
                    : (s.ID.GetType().GetProperty("ID")?.GetValue(s.ID, null)?.ToString() ?? s.ID.ToString());
            }
            catch { /* ignore */ }

            if (!string.IsNullOrWhiteSpace(code))
                parts.Add($"Code: {code}");
            if (!string.IsNullOrWhiteSpace(s.Name))
                parts.Add($"Name: {s.Name}");

            //email
            var email = GetSellerContactEmail(invoice);
            if (!string.IsNullOrWhiteSpace(email))
                parts.Add($"EmailAddress: {email}");

            //contact person
            var contactName = invoice?.SellerContact?.Name;
            if (!string.IsNullOrWhiteSpace(contactName))
                parts.Add($"ContactPerson: {contactName}");

            //address
            if(!string.IsNullOrWhiteSpace(s.Street))
                parts.Add($"Street: {s.Street}");
            if(!string.IsNullOrWhiteSpace(s.Postcode))
                parts.Add($"ZipCode: {s.Postcode}");
            if(!string.IsNullOrWhiteSpace(s.City))
                parts.Add($"City: {s.City}");
            var country = s.Country?.ToString();
            if(!string.IsNullOrWhiteSpace(country))
                parts.Add($"Country: {country}");         

            //tax nr. (FC) / USt-IdNr. (VA)
            var vatId = invoice?.SellerTaxRegistration?.FirstOrDefault(r => r.SchemeID == TaxRegistrationSchemeID.VA)?.No;
            var taxNo = invoice?.SellerTaxRegistration?.FirstOrDefault(r => r.SchemeID == TaxRegistrationSchemeID.FC)?.No;

            if(!string.IsNullOrWhiteSpace(vatId))
                parts.Add($"VatIDNum: {vatId}");
            if(!string.IsNullOrWhiteSpace(taxNo))
                parts.Add($"FederalTaxID: {taxNo}");
                
            return string.Join(" | ", parts);
        }

        //help method for getting Mail from SellerContact (regardless of spelling)
        static string? GetSellerContactEmail(InvoiceDescriptor invoice)
        {
            var contact = invoice?.SellerContact;
            if (contact == null) return null;

            var type = contact.GetType();
            var property = type.GetProperty("EmailAddress")
                ?? type.GetProperty("EMailAddress")
                ?? type.GetProperty("Email");

            return property?.GetValue(contact, null)?.ToString();
        }

        //tax help methods
        static string? GetVatID(InvoiceDescriptor invoice)
        {
            //VAT-ID = USt-IdNr.
            var vatId = invoice?.SellerTaxRegistration?.FirstOrDefault(r => r.SchemeID == TaxRegistrationSchemeID.VA)?.No;
            return NormalizeVat(vatId);
        }

        static string? GetTaxNr(InvoiceDescriptor invoice)
        {
            //FC = Fiscal Code = nationale Steuernummer
            var taxNr = invoice.SellerTaxRegistration?.FirstOrDefault(r => r.SchemeID == TaxRegistrationSchemeID.FC)?.No;
            return NormalizeId(taxNr);
        }

        //clear text help methods
        static string? NormalizeVat(string? s)
        {
            if (string.IsNullOrWhiteSpace(s)) 
                return null;

            s = s.Replace(" ", "").Replace("-", "").ToUpperInvariant();
            return s;
        }

        static string? NormalizeId(string? s)
        {
            if (string.IsNullOrWhiteSpace(s))
                return null;

            s = s.Trim();
            s = Regex.Replace(s, @"\s|-|\.", ""); //remove space/dots/dash
            return s.ToUpperInvariant();
        }

        static string NormalizeName(string s)
        {
            s = s.ToLowerInvariant();
            s = Regex.Replace(s, @"[^\p{L}\p{N}\s]", " ");
            s = Regex.Replace(s, @"\b(gmbh|ag|kg|mbh|co|kg|ohg|ug|limited|ltd|sarl|srl|spa|bv|nv|ab|oy|as|aps|kft|sro|spzoo|oü)\b", " ");
            s = Regex.Replace(s, @"\s+", " ").Trim();
            return s;
        }

        static string Normalize(string s)
        {
            s = Regex.Replace(s.Trim().ToLowerInvariant(), @"\s+", " ");
            return s;
        }

        //compare help methods
        static bool SameVat(string? a, string? b)
        {
            a = NormalizeVat(a);
            b = NormalizeVat(b);
            return !string.IsNullOrWhiteSpace(a) && a == b;
        }

        static bool SameId(string? a, string? b)
        {
            a = NormalizeId(a);
            b = NormalizeId(b);
            return !string.IsNullOrWhiteSpace(a) && a == b;
        }

        static bool SameText(string? a, string? b)
        {
            a = Normalize(a);
            b = Normalize(b ?? "");
            return string.Equals(a, b, StringComparison.Ordinal);
        }

        //compares 2 values and returns 0-1
        static double CosineSimilarity(IReadOnlyList<float> v1, IReadOnlyList<float> v2)
        {
            double dot = 0;
            double n1 = 0;
            double n2 = 0;
            int length = Math.Min(v1.Count, v2.Count);

            for (int i = 0; i < length; i++)
            {
                dot += v1[i] * v2[i]; 
                n1  += v1[i] * v1[i];
                n2  += v2[i] * v2[i];
            }

            if (n1 == 0 || n2 == 0)
            {
                return 0;
            }
            else
            {
                return dot / (Math.Sqrt(n1) * Math.Sqrt(n2));
            }
        }

        //Jaccard similarity index (value between 0 and 1)
        static double NameSimilarityIndex(string a, string b)
        {
            var A = new HashSet<string>(a.Split(' ', StringSplitOptions.RemoveEmptyEntries));
            var B = new HashSet<string>(b.Split(' ', StringSplitOptions.RemoveEmptyEntries));
            if (A.Count == 0 && B.Count == 0)
                return 1.0;

            var inter = A.Intersect(B).Count();
            var union = A.Union(B).Count();

            if (union == 0)
            {
                return 0;
            }
            else
            {
                return (double)inter / union; //calculates the sum of same words in relation to the sum of different words
            }
        }

        //First City/Country help methods (returns FirstOrDefault)
        static string? FirstCity(VendorVectorItem v)
        {
            string? city = v.BPAddresses?.FirstOrDefault()?.City;
            return city;
        }

        static string? FirstCountry(VendorVectorItem v)
        {
            string? country = v.BPAddresses?.FirstOrDefault()?.Country;
            return country;
        }

        //extracting available SupplierIds
        public static List<string> ExtractSupplierIds(InvoiceDescriptor invoice)
        {
            List<string> ids = new();

            void Add(string? id)
            {
                if(!string.IsNullOrWhiteSpace(id))
                {
                    ids.Add(id.Trim().ToLowerInvariant());
                }
            }

            if(invoice?.Seller != null)
            {
                Add(invoice.Seller.ID?.ID);

                Add(invoice.Seller.GlobalID?.ID);

                Add(invoice.Seller.SpecifiedLegalOrganization?.ID?.ID);
            }

            /*foreach(var line in invoice?.TradeLineItems ?? Enumerable.Empty<TradeLineItem>())
            {
                Add(line.SellerAssignedID);
            }*/

            return ids.Distinct().ToList();
        }
    }
}