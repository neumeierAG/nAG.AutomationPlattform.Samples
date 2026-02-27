using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel.ChatCompletion;
using s2industries.ZUGFeRD;

namespace AiImportVendor
{
    public static class ItemAiHelper
    {
        private const double SupplierCatalogBoost = 0.50;
        private const double BarcodeBoost = 0.50;
        private const double BuyerIdBoost = 0.30;
        private const double NameBoostHigh = 0.20;
        private const double NameBoostMedium = 0.10;
        private const double NameBoostLow = 0.05;
        private const double NameSimilarityHighThreshold = 0.90;
        private const double NameSimilarityMediumThreshold = 0.75;
        private const double NameSimilarityLowThreshold = 0.50;
        private const double CosineWeight = 0.70;
        private const double AcceptanceThreshold = 0.85;
        private const double AcceptanceFloor = 0.75;
        private const double AcceptanceDelta = 0.10;

        /// <summary>
        /// Gets the best matching SAP item code for a billing line via AI.
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
            logger ??= static (_, _) => Task.CompletedTask;

            try
            {
                if (!TryValidateInputs(line, items, itemEmbeddings, out var validationMessage))
                {
                    await logger($"ITM-Match failed: {validationMessage}", Backend.Database.RequestStatus.Warning);
                    return fallbackItemCode;
                }

                var candidateItems = GetCandidateItems(items, itemEmbeddings);
                if (candidateItems.Count == 0)
                {
                    await logger("ITM-Match failed: no candidate items with embeddings.", Backend.Database.RequestStatus.Warning);
                    return fallbackItemCode;
                }

                var (sellerIds, buyerId, barcodes, nameTokens) = ExtractLineFeatures(line);

                string lineSemantic = ToSemanticStringLine(line);
                if (string.IsNullOrWhiteSpace(lineSemantic))
                {
                    lineSemantic = $"{line.BuyerAssignedID} | {line.SellerAssignedID} | {line.Name}";
                }

                await logger($"KI-ITM-Info: {lineSemantic}", Backend.Database.RequestStatus.Success);
                var queryEmbedding = await embeddingService.GenerateAsync(lineSemantic);
                var queryVector = queryEmbedding?.Vector.ToArray();

                if (queryVector == null || queryVector.Length == 0)
                {
                    await logger("ITM-Match failed: query embedding is empty. Using fallback.", Backend.Database.RequestStatus.Warning);
                    return fallbackItemCode;
                }

                var scored = ScoreCandidates(candidateItems, itemEmbeddings, queryVector, sellerIds, buyerId, barcodes, nameTokens);
                var ordered = scored.OrderByDescending(x => x.Score).ToList();

                if (TryResolveByScore(ordered, out var bestItemCode, out var bestScore, out var delta, out var bestReason))
                {
                    await logger(
                        $"ITM-Match success {bestItemCode}: score={bestScore:F2} delta={delta:F2} [{bestReason}]",
                        Backend.Database.RequestStatus.Success);
                    return bestItemCode!;
                }

                var top3 = ordered.Take(3).ToList();
                var llmPick = await TryResolveByLlmAsync(top3, candidateItems, lineSemantic, fallbackItemCode, chatService);
                if (!string.IsNullOrWhiteSpace(llmPick))
                {
                    await logger($"ITM-Match (LLM) success -> Pick: {llmPick}", Backend.Database.RequestStatus.Success);
                    return llmPick;
                }

                await logger("ITM-Match failed: using final fallback.", Backend.Database.RequestStatus.Warning);
                return fallbackItemCode;
            }
            catch (Exception ex)
            {
                await logger($"ITM-Match exception: {ex.GetType().Name}: {ex.Message}.", Backend.Database.RequestStatus.Warning);
                return fallbackItemCode;
            }
        }

        private static bool TryValidateInputs(
            TradeLineItem? line,
            List<ItemVectorItem>? items,
            Dictionary<string, Embedding<float>>? itemEmbeddings,
            out string message)
        {
            if (line == null)
            {
                message = "line is null";
                return false;
            }

            if (items == null || items.Count == 0)
            {
                message = "items are empty";
                return false;
            }

            if (itemEmbeddings == null || itemEmbeddings.Count == 0)
            {
                message = "item embeddings are empty";
                return false;
            }

            message = string.Empty;
            return true;
        }

        private static List<ItemVectorItem> GetCandidateItems(
            IEnumerable<ItemVectorItem> items,
            IReadOnlyDictionary<string, Embedding<float>> itemEmbeddings)
        {
            return items
                .Where(i => !string.IsNullOrWhiteSpace(i.ItemCode) && itemEmbeddings.ContainsKey(i.ItemCode!))
                .ToList();
        }

        private static (HashSet<string> SellerIds, string? BuyerId, HashSet<string> Barcodes, string NameTokens) ExtractLineFeatures(
            TradeLineItem line)
        {
            return (
                GetSupplierIds(line),
                NormalizeId(line.BuyerAssignedID),
                GetBarcodes(line),
                NormalizeName(line.Name) ?? string.Empty
            );
        }

        private static List<(string ItemCode, double Score, string Reason)> ScoreCandidates(
            IEnumerable<ItemVectorItem> candidateItems,
            IReadOnlyDictionary<string, Embedding<float>> itemEmbeddings,
            IReadOnlyList<float> queryVector,
            IReadOnlySet<string> sellerIds,
            string? buyerId,
            IReadOnlySet<string> barcodes,
            string nameTokens)
        {
            var scored = new List<(string ItemCode, double Score, string Reason)>();

            foreach (var item in candidateItems)
            {
                var itemCode = item.ItemCode!;
                if (!itemEmbeddings.TryGetValue(itemCode, out var embedding) || embedding?.Vector == null || embedding.Vector.Length == 0)
                {
                    continue;
                }

                double cosine = CosineSimilarity(queryVector, embedding.Vector.ToArray());
                double boost = 0.0;
                var reasons = new List<string> { $"cos={cosine:F3}" };

                if (!string.IsNullOrWhiteSpace(item.SupplierCatalogNo) && sellerIds.Contains(NormalizeId(item.SupplierCatalogNo)!))
                {
                    boost += SupplierCatalogBoost;
                    reasons.Add($"SupplierCatalogNo=match(+{SupplierCatalogBoost:F2})");
                }

                if (!string.IsNullOrWhiteSpace(item.Barcode) && barcodes.Contains(NormalizeBarcode(item.Barcode)!))
                {
                    boost += BarcodeBoost;
                    reasons.Add($"Barcode=match(+{BarcodeBoost:F2})");
                }

                if (!string.IsNullOrWhiteSpace(buyerId) && SameId(buyerId, item.ItemCode))
                {
                    boost += BuyerIdBoost;
                    reasons.Add($"BuyerId<>ItemCode=match(+{BuyerIdBoost:F2})");
                }

                double nameSimilarity = Math.Max(
                    NameSimilarityIndex(nameTokens, NormalizeName(item.ItemName) ?? string.Empty),
                    Math.Max(
                        NameSimilarityIndex(nameTokens, NormalizeName(item.ForeignName) ?? string.Empty),
                        NameSimilarityIndex(nameTokens, NormalizeName(item.ItemGroups?.GroupName) ?? string.Empty)
                    )
                );

                if (nameSimilarity >= NameSimilarityHighThreshold)
                {
                    boost += NameBoostHigh;
                    reasons.Add($"Name~{nameSimilarity:F2}(+{NameBoostHigh:F2})");
                }
                else if (nameSimilarity >= NameSimilarityMediumThreshold)
                {
                    boost += NameBoostMedium;
                    reasons.Add($"Name~{nameSimilarity:F2}(+{NameBoostMedium:F2})");
                }
                else if (nameSimilarity >= NameSimilarityLowThreshold)
                {
                    boost += NameBoostLow;
                    reasons.Add($"Name~{nameSimilarity:F2}(+{NameBoostLow:F2})");
                }

                double finalScore = Math.Min(1.0, (CosineWeight * cosine) + boost);
                scored.Add((itemCode, finalScore, string.Join(", ", reasons)));
            }

            return scored;
        }

        private static bool TryResolveByScore(
            IReadOnlyList<(string ItemCode, double Score, string Reason)> ordered,
            out string? itemCode,
            out double score,
            out double delta,
            out string? reason)
        {
            var best = ordered.FirstOrDefault();
            var second = ordered.Skip(1).FirstOrDefault();

            if (best.ItemCode == null)
            {
                itemCode = null;
                score = 0;
                delta = 0;
                reason = null;
                return false;
            }

            delta = best.Score - (second.ItemCode == null ? 0 : second.Score);
            bool accepted = best.Score >= AcceptanceThreshold || (best.Score >= AcceptanceFloor && delta >= AcceptanceDelta);

            itemCode = accepted ? best.ItemCode : null;
            score = best.Score;
            reason = best.Reason;
            return accepted;
        }

        private static async Task<string?> TryResolveByLlmAsync(
            IReadOnlyList<(string ItemCode, double Score, string Reason)> topCandidates,
            IEnumerable<ItemVectorItem> candidateItems,
            string lineSemantic,
            string fallbackItemCode,
            IChatCompletionService chatService)
        {
            if (topCandidates.Count == 0)
            {
                return null;
            }

            var candidateMap = candidateItems
                .Where(i => !string.IsNullOrWhiteSpace(i.ItemCode))
                .GroupBy(i => i.ItemCode!)
                .ToDictionary(g => g.Key, g => g.First());

            string topText = string.Join("\n", topCandidates.Select((candidate, index) =>
            {
                if (!candidateMap.TryGetValue(candidate.ItemCode, out var item))
                {
                    return $"{index + 1}. {candidate.ItemCode}";
                }

                return $"{index + 1}. {candidate.ItemCode}: {item.ToSemanticString()}";
            }));

            string prompt = $"""
                There is a billing line (semantic summary):
                "{lineSemantic}"

                These are the 3 most similar SAP articles (ItemCode | ItemName | ForeignName | Barcode | SupplierCatalogNo | GroupName):
                {topText}

                Return only the ItemCode of the single best match.
                If there is no suitable match, return '{fallbackItemCode}'.
                """;

            var llm = await chatService.GetChatMessageContentAsync(prompt);
            var pick = llm?.Content?.Trim();

            return string.IsNullOrWhiteSpace(pick) ? null : pick;
        }

        private static string ToSemanticStringLine(TradeLineItem line)
        {
            var parts = new List<string>();
            if (!string.IsNullOrWhiteSpace(line?.SellerAssignedID))
            {
                parts.Add($"ItemCode: {line.SellerAssignedID}");
            }

            if (!string.IsNullOrWhiteSpace(line?.Name))
            {
                parts.Add($"Name: {line.Name}");
            }

            var foreignName = line?.AssociatedDocument?.Notes?.FirstOrDefault()?.Content;
            if (!string.IsNullOrWhiteSpace(foreignName))
            {
                parts.Add($"Fremdname: {foreignName}");
            }

            var barcode = line?.DesignatedProductClassifications?.FirstOrDefault()?.ClassName;
            if (!string.IsNullOrWhiteSpace(barcode))
            {
                parts.Add($"Barcode: {barcode}");
            }

            if (!string.IsNullOrWhiteSpace(line?.GlobalID?.ID))
            {
                parts.Add($"Lieferanten-Nr.: {line.GlobalID.ID}");
            }

            if (!string.IsNullOrWhiteSpace(line?.BuyerAssignedID))
            {
                parts.Add($"Produktgruppe: {line.BuyerAssignedID}");
            }

            return string.Join(" | ", parts);
        }

        private static HashSet<string> GetSupplierIds(TradeLineItem line)
        {
            var set = new HashSet<string>(StringComparer.Ordinal);
            if (!string.IsNullOrWhiteSpace(line?.SellerAssignedID))
            {
                set.Add(NormalizeId(line.SellerAssignedID)!);
            }
            if (!string.IsNullOrWhiteSpace(line?.GlobalID?.ID))
            {
                set.Add(NormalizeId(line.GlobalID.ID)!);
            }
            return set;
        }

        private static HashSet<string> GetBarcodes(TradeLineItem line)
        {
            var set = new HashSet<string>(StringComparer.Ordinal);
            var firstBarcode = line?.DesignatedProductClassifications?.FirstOrDefault()?.ClassName;
            if (!string.IsNullOrWhiteSpace(firstBarcode))
            {
                set.Add(NormalizeBarcode(firstBarcode)!);
            }
            return set;
        }

        private static string? NormalizeId(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return null;
            }

            value = value.Trim();
            value = Regex.Replace(value, @"\s|-|\.", string.Empty);
            return value.ToUpperInvariant();
        }

        private static string? NormalizeBarcode(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                return null;
            }

            value = Regex.Replace(value, @"\s|-", string.Empty);
            return value.ToUpperInvariant();
        }

        private static string? NormalizeName(string? value)
        {
            value = (value ?? string.Empty).ToLowerInvariant();
            value = Regex.Replace(value, @"[^\p{L}\p{N}\s]", " ");
            value = Regex.Replace(value, @"\s+", " ").Trim();
            return value;
        }

        private static bool SameId(string? left, string? right)
        {
            left = NormalizeId(left);
            right = NormalizeId(right);
            return !string.IsNullOrWhiteSpace(left) && left == right;
        }

        private static double CosineSimilarity(IReadOnlyList<float> v1, IReadOnlyList<float> v2)
        {
            double dot = 0;
            double n1 = 0;
            double n2 = 0;
            int length = Math.Min(v1.Count, v2.Count);

            for (int i = 0; i < length; i++)
            {
                dot += v1[i] * v2[i];
                n1 += v1[i] * v1[i];
                n2 += v2[i] * v2[i];
            }

            if (n1 == 0 || n2 == 0)
            {
                return 0;
            }

            return dot / (Math.Sqrt(n1) * Math.Sqrt(n2));
        }

        private static double NameSimilarityIndex(string a, string b)
        {
            var setA = new HashSet<string>(a.Split(' ', StringSplitOptions.RemoveEmptyEntries));
            var setB = new HashSet<string>(b.Split(' ', StringSplitOptions.RemoveEmptyEntries));
            if (setA.Count == 0 && setB.Count == 0)
            {
                return 1.0;
            }

            int inter = setA.Intersect(setB).Count();
            int union = setA.Union(setB).Count();
            if (union == 0)
            {
                return 0;
            }

            return (double)inter / union;
        }
    }
}
