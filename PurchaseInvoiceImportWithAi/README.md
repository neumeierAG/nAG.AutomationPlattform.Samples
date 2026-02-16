## Description

Automation for the import of electronic invoices. An e-invoice gets transfered via Filedrop as XRechnung or ZUGFeRD.
The automation identifies the format and creates a draft invoice in SAP Business One and attatches the original file under attachments.

## Identification of business partners and items through AI-embedding

    1. Clear result with sufficient distance from the next business partner / item.
    2. If there is no clear result after step 1., the AI selects the best match by comparing the 3 highest scores.
    3. If step 2. also fails to produce a clear result, a predetermed fallback value will be selected.