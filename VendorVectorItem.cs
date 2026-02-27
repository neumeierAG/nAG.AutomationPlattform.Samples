using System.Text.Json.Serialization;
using System.Collections.Generic;

namespace AiImportVendor 
{
    public class VendorVectorItem 
    {
        public string? CardCode { get; set; }
        public string? CardName { get; set; }
        public string? EmailAddress { get; set; }
        public string? ContactPerson { get; set; }
        [JsonPropertyName("Address")]
        public string? Street { get; set; }
        public string? ZipCode { get; set; }
        public string? City { get; set; }
        public string? Country { get; set; }
        public string? HouseBankIBAN { get; set; }
        public string? FederalTaxID { get; set; }
        public string? VatIDNum { get; set; }
        public string? SupplierCatalogNr { get; set; }
        
        public List<BPAddress>? BPAddresses { get; set; }

        //subclass for saving addresses
        public class BPAddress
        {
            public string? AddressName { get; set; }
            public string? AddressType { get; set; }
            public string? Street { get; set; }
            public string? ZipCode { get; set; }
            public string? City { get; set; }
            public string? Country { get; set; }
        }

        //help method for readability (CardCode | CardName | ...)
        public string ToSemanticString()
        {
            var parts = new List<string>();

            if (!string.IsNullOrWhiteSpace(CardCode))
                parts.Add($"Code: {CardCode}");
            if (!string.IsNullOrWhiteSpace(CardName))
                parts.Add($"Name: {CardName}");
            if (!string.IsNullOrWhiteSpace(EmailAddress))
                parts.Add($"EmailAddress: {EmailAddress}");
            if (!string.IsNullOrWhiteSpace(ContactPerson))
                parts.Add($"ContactPerson: {ContactPerson}");
                
            if (!string.IsNullOrWhiteSpace(Street))
                parts.Add($"Street: {Street}");
            if (!string.IsNullOrWhiteSpace(ZipCode))
                parts.Add($"ZipCode: {ZipCode}");
            if (!string.IsNullOrWhiteSpace(City))
                parts.Add($"City: {City}");
            if (!string.IsNullOrWhiteSpace(Country))
                parts.Add($"Country: {Country}");
                
            if (!string.IsNullOrWhiteSpace(FederalTaxID))
                parts.Add($"FederalTaxID: {FederalTaxID}");
            if (!string.IsNullOrWhiteSpace(VatIDNum))
                parts.Add($"VatIDNum: {VatIDNum}");
            if (!string.IsNullOrWhiteSpace(SupplierCatalogNr))
                parts.Add($"SupplierCatalogNr: {SupplierCatalogNr}");

            return string.Join(" | ", parts);
        }
    }
}