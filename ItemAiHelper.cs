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
    public static class ItemAiHelper
    {
        /// <summary>
        /// Gets the best matching SAP item code for a billing line via AI
        /// </summary>
        public static async Task<string> ResolveItemCodeAsync(
            TradeLineItem line,
            List<ItemVectorItem> items,
            Dictionary<string, Embedding<float>> itemEmbeddings,
            IEmbeddingGenerator<string, Embedding<float>> embeddingService,
            IChatCompletionService chatService,
            string fallbackItemCode = "E10000",
            Func<string, Backend.Database.RequestStatus, Task>? logger = null)
        {
            try
            {
                //fallbacks before calculation
                if(line == null)
                {
                    if(logger != null) await logger("❌ITM-Match failed → line == null", Backend.Database.RequestStatus.Warning);
                    return fallbackItemCode;
                }
                if(items == null || items.Count == 0)
                {
                    if(logger != null) await logger("❌ITM-Match failed → items leer", Backend.Database.RequestStatus.Warning);
                    return fallbackItemCode;
                }
                if(itemEmbeddings == null || itemEmbeddings.Count == 0)
                {
                    if(logger != null) await logger("❌ITM-Match failed → itemEmbeddings leer", Backend.Database.RequestStatus.Warning);
                    return fallbackItemCode;
                }

                //validate if item has corresponding embedding; fallback if not
                var candidateItems = items
                                        .Where(i => !string.IsNullOrWhiteSpace(i.ItemCode) && itemEmbeddings.ContainsKey(i.ItemCode!))
                                        .ToList();
                if (candidateItems.Count == 0)
                {
                    if (logger != null) await logger("❌ITM-Match failed → keine Kandidaten mit Embedding.", Backend.Database.RequestStatus.Warning);
                    return fallbackItemCode;
                }

                //1)
                //unique line values
                var sellerIds   = GetSupplierIds(line);
                var buyerId     = NormalizeId(line?.BuyerAssignedID);
                var barcodes    = GetBarcodes(line);
                var nameTokens  = NormalizeName(line?.Name ?? "");

                //2)
                // create item embeddings
                string lineSemantic = ToSemanticStringLine(line);
                if (string.IsNullOrWhiteSpace(lineSemantic))
                {
                    lineSemantic = $"{line?.BuyerAssignedID} | {line?.SellerAssignedID} | {line?.Name}";
                }

                await logger($"KI-ITM-Info: {lineSemantic}", Backend.Database.RequestStatus.Success);
                var queryEmbedding = await embeddingService.GenerateAsync(lineSemantic);

                var queryVector = queryEmbedding?.Vector.ToArray();
                if (queryVector == null || queryVector.Length == 0)
                {
                    if (logger != null)
                        await logger("❌ITM-Match failed → Querry Embedding leer → Fallback", Backend.Database.RequestStatus.Warning);
                    
                    return fallbackItemCode;
                }

                //3)
                //line score calculation
                var scored = new List<(string ItemCode, double Score, string Reason)>();

                foreach (var it in candidateItems)
                {
                    var code = it.ItemCode!;
                    if (!itemEmbeddings.TryGetValue(code, out var emb) || emb?.Vector == null || emb.Vector.Length == 0)
                        continue;

                    double cosine = CosineSimilarity(queryVector, emb.Vector.ToArray());
                    double boost = 0.0;
                    var reasons = new List<string> { $"cos={cosine:F3}" };

                    //Supplier-catalognumber and seller-IDs match (highscore)
                    if (!string.IsNullOrWhiteSpace(it.SupplierCatalogNo) && sellerIds.Contains(NormalizeId(it.SupplierCatalogNo)))
                    {
                        boost += 0.50;
                        reasons.Add("SupplierCatalogNr=match(+0.50)");
                    }

                    //Barcode/EAN match (highscore)
                    if (!string.IsNullOrWhiteSpace(it.Barcode) && barcodes.Contains(NormalizeBarcode(it.Barcode)))
                    {
                        boost += 0.50;
                        reasons.Add("Barcode=match(0.50)");
                    }

                    //BuyerAssignedID and ItemCode match (middle-score)
                    if(!string.IsNullOrWhiteSpace(buyerId) && SameId(buyerId, it.ItemCode))
                    {
                        boost += 0.30;
                        reasons.Add("BuyerId<>ItemCode=match(+0.30)");
                    }

                    //similar names (ItemName / ForeignName / GroupName) (low-middle score)
                    double nameSimilar = Math.Max(
                                                NameSimilarityIndex(nameTokens, NormalizeName(it.ItemName ?? "")), 
                                                Math.Max(
                                                    NameSimilarityIndex(nameTokens, NormalizeName(it.ForeignName ?? "")),
                                                    NameSimilarityIndex(nameTokens, NormalizeName(it.ItemGroups?.GroupName ?? ""))
                                                )
                    );
                    if (nameSimilar >= 0.9)
                    {
                        boost +=0.20;
                        reasons.Add($"Name≈{nameSimilar:F2}(+0.20)");
                    }
                    else if (nameSimilar >= 0.75)
                    {
                        boost +=0.10;
                        reasons.Add($"Name≈{nameSimilar:F2}(+0.10)");
                    }
                    else if (nameSimilar >= 0.5)
                    {
                        boost +=0.05;
                        reasons.Add($"Name≈{nameSimilar:F2}(+0.05)");
                    }

                    //final score = 70% cosine + booster
                    double final = Math.Min(1.0, (0.70 * cosine) + boost);
                    scored.Add((code, final, string.Join(", ", reasons)));
                }

                //sort based on score
                var ordered = scored.OrderByDescending(x => x.Score).ToList();
                var best = ordered.FirstOrDefault();
                var second = ordered.Skip(1).FirstOrDefault();

                //4)
                // threshholds; either highscore or clear difference to 2nd best
                if (best.ItemCode != null)
                {
                    double delta = best.Score - (second.ItemCode == null ? 0 : second.Score);

                    if (best.Score >= 0.85 || (best.Score >= 075 && delta >= 0.10))
                    {
                        if (logger != null) await logger($"✅ITM-Match success {best.ItemCode}: score={best.Score:F2} Δ= {delta:F2} [{best.Reason}]", Backend.Database.RequestStatus.Success);
                        return best.ItemCode;
                    }
                }

                //5)
                //LLM-Fallback: Ask Ai for top 3
                var top3 = ordered.Take(3).ToList();
                if (top3.Count > 0)
                {
                    //english for token efficiency and performance
                    string prompt = $@"
                        There is a billing line (semantic summary):
                        ""{lineSemantic}""
                        
                        That are the 3 most similar articles from SAP (ItemCode | ItemName | ForeignName | Barcode | SupplierCatalogNr | Groupname):
                        {string.Join("\n", top3.Select((m, i) => $"{i + 1}. {m.ItemCode}: {(candidateItems.First(itm => itm.ItemCode == m.ItemCode)).ToSemanticString()}"))}
                        
                        Only return the ItemCode of the single best match.
                        If there is none, return '{fallbackItemCode}'.
                        ";
                    var llm = await chatService.GetChatMessageContentAsync(prompt);
                    var pick = llm?.Content?.Trim();

                    if (!string.IsNullOrWhiteSpace(pick))
                    {
                        if (logger != null)
                        {
                            await logger($@"✅ITM-Match (LLM) success → Pick: {pick}
                                (Top3: {string.Join(">> ", top3.Select((t, i) => $"<< #{i + 1}: {(candidateItems.First(itm => itm.ItemCode == t.ItemCode)).ToSemanticString()}"))})",
                                Backend.Database.RequestStatus.Success);
                            return pick;
                        }
                    }
                }

                //6)
                //final fallback
                if (logger != null) await logger("❌ITM-Match failed → Letzter Fallback verwendet.", Backend.Database.RequestStatus.Warning);
                return fallbackItemCode;
            }
            catch (Exception ex)
            {
                if (logger != null) await logger($"❌ITM-Match Exception: {ex.GetType().Name}: {ex.Message}.", Backend.Database.RequestStatus.Warning);
                return fallbackItemCode;
            }
        }

        ///////////////////////////////////////////////////////// Help Methods //////////////////////////////////////////////////////////////
        /// 
        //help methods for readability (item information | ...)
        private static string ToSemanticStringLine(TradeLineItem line)
        {
            var parts = new List<string>();
            //item code
            if (!string.IsNullOrWhiteSpace(line?.SellerAssignedID)) 
                parts.Add($"ItemCode: {line.SellerAssignedID}");

            //item name
            if (!string.IsNullOrWhiteSpace(line?.Name))
                parts.Add($"Name: {line.Name}");

            //foreign name
            var foreignName = line?.AssociatedDocument?.Notes?.FirstOrDefault()?.Content;
            if (!string.IsNullOrWhiteSpace(foreignName))
                parts.Add($"Fremdname: {foreignName}");

            //barcode
            var barcode = line?.DesignatedProductClassifications?.FirstOrDefault()?.ClassName;
            if (!string.IsNullOrWhiteSpace(barcode))
                parts.Add($"Barcode: {barcode}");

            //supplier nr.
            if (!string.IsNullOrWhiteSpace(line?.GlobalID?.ID))
                parts.Add($"Lieferanten-Nr.: {line.GlobalID.ID}");

            //product group
            if (!string.IsNullOrWhiteSpace(line?.BuyerAssignedID))
                parts.Add($"Produktgruppe: {line.BuyerAssignedID}");

            return string.Join(" | ", parts);
        }

        //item info help methods
        private static HashSet<string> GetSupplierIds(TradeLineItem line)
        {
            var set = new HashSet<string>(StringComparer.Ordinal);
            if (!string.IsNullOrWhiteSpace(line?.SellerAssignedID))
            {
                set.Add(NormalizeId(line.SellerAssignedID)!);
            }
            if (!string.IsNullOrWhiteSpace(line?.GlobalID?.ID))
            {
                set.Add(NormalizeId(line?.GlobalID?.ID)!);
            }
            return set;
        }

        private static HashSet<string> GetBarcodes(TradeLineItem line)
        {
            var set = new HashSet<string>(StringComparer.Ordinal);
            var firstBc = line?.DesignatedProductClassifications?.FirstOrDefault()?.ClassName;
            if (!string.IsNullOrWhiteSpace(firstBc))
            {
                set.Add(NormalizeId(firstBc)!);
            }
            return set;
        }

        //clear text help methods
        private static string? NormalizeId(string? s)
        {
            if (string.IsNullOrWhiteSpace(s))
                return null;

            s = s.Trim();
            s = Regex.Replace(s, @"\s|-|\.", ""); //remove space/dots/dash
            return s.ToUpperInvariant();
        }

        private static string? NormalizeBarcode(string? s)
        {
            if (string.IsNullOrWhiteSpace(s))
                return null;

            s = Regex.Replace(s, @"\s|-", ""); //only numbers and letters
            return s.ToUpperInvariant();
        }

        private static string? NormalizeName(string? s)
        {
            s = (s ?? "").ToLowerInvariant();
            s = Regex.Replace(s, @"[^\p{L}\p{N}\s]", " ");
            s = Regex.Replace(s, @"\s+", " ").Trim(); 
            return s;
        }

        //compare help methods
        private static bool SameId(string? a, string? b)
        {
            a = NormalizeId(a);
            b = NormalizeId(b);
            return !string.IsNullOrWhiteSpace(a) && a == b;
        }

        //compares 2 values and returns 0< - >1
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
    }
}
