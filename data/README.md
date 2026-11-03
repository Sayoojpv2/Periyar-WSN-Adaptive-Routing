# Input dataset (not included)

The simulations read one Excel workbook, `data/consdata.xlsx`. It is **not part of this repository**:
it is an analysis report of the Kerala State Pollution Control Board (KSPCB), and no licence permitting
its redistribution has been established.

## What the file is

* KSPCB, Environmental Surveillance Centre, Eloor — *Analysis Report of River Periyar, January–July 2025*.
* One sheet per sampling station (10 sheets): VETTUKADAVU, PATHALAM BRIDGE, PUTHALAM KADAVU,
  PATHALAM BUND US, PATHALAM BUNDS DS, MANJUMMAL BUND, PURALLIKADAVU US, PURALLIKADAVU DS, MANAPPURAM,
  PURAPPILLIKADAVU BUND NAVIGATIO.
* Size 1,208,612 bytes; MD5 `6100b72899bd76df5a393fbbc5fb898a` (the file used for the published results).

## How to set it up

1. Obtain the report workbook from KSPCB (Environmental Surveillance Centre, Eloor, or the KSPCB
   office/website) or from the project author.
2. Save it as `data/consdata.xlsx` inside this project folder. The path is set in `v2_config.m`
   (`cfg.workbook`); no other location is used.
3. Check that it is the same file (MATLAB):

   ```matlab
   md5 = lower(reshape(dec2hex(typecast(java.security.MessageDigest.getInstance('MD5').digest( ...
       fread(fopen('data/consdata.xlsx'), inf, '*uint8')), 'uint8'))', 1, []))
   % expected: 6100b72899bd76df5a393fbbc5fb898a
   ```

A different version of the workbook will change the workload and therefore the results.
