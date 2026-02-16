using System.Collections.Generic;
using nAG.Framework.SapBusinessOne.Models;

namespace AiImportVendor
{
    public class ItemVectorItem 
    {
        public string? ItemCode { get; set; }
        public string? ItemName { get; set; }
        public string? ForeignName { get; set; }
        public string? Barcode { get; set; }
        public string? SupplierCatalogNo { get; set; }
        public ItemGroups? ItemGroups { get; set; }

        //subclass ItemGroup
        public class ItemGroup
        {
            public string? GroupName { get; set; }
        }

        //help method for readability (ItemCode | ItemName | ...)
        public string ToSemanticString()
        {
            var parts = new List<string>();

            if (!string.IsNullOrWhiteSpace(ItemCode))
                parts.Add($"Code: {ItemCode}");
            if (!string.IsNullOrWhiteSpace(ItemName))
                parts.Add($"Name: {ItemName}");
            if (!string.IsNullOrWhiteSpace(ForeignName))
                parts.Add($"Fremdname: {ForeignName}");
            if (!string.IsNullOrWhiteSpace(Barcode))
                parts.Add($"Barcode: {Barcode}");
            if (!string.IsNullOrWhiteSpace(SupplierCatalogNo))
                parts.Add($"Lieferant-Nr: {SupplierCatalogNo}");
            if (!string.IsNullOrWhiteSpace(ItemGroups?.GroupName))
                parts.Add($"Produktgruppe: {ItemGroups?.GroupName}");

            return string.Join(" | ", parts);
        }
    }
}
