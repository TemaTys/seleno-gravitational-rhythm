# Semantic Map: Crime Categorization

This document outlines the mapping of raw crime descriptions from various city databases into unified analytical families for the SGR system.

| Semantic Family | Chicago | Philadelphia (Philly) | New York (NYC) | Los Angeles (LA) | San Francisco (SF) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`AGGRESSION_TOTAL`** | `ASSAULT`<br>`BATTERY`<br>`HOMICIDE` | `Aggravated Assault Firearm`<br>`Aggravated Assault No Firearm`<br>`Homicide - Criminal` | `FELONY ASSAULT`<br>`ASSAULT 3 & RELATED OFFENSES`<br>`MURDER & NON-NEGL. MANSLAUGHTER` | `ASSAULT WITH DEADLY WEAPON AGGRAVATED ASSAULT`<br>`BATTERY - SIMPLE ASSAULT`<br>`CRIMINAL HOMICIDE` | `ASSAULT` |
| **`SEXUAL_ASSAULT_CORE`** | `CRIM SEXUAL ASSAULT` | `Rape` | `RAPE` | `RAPE FORCIBLE` | `SEX OFFENSES FORCIBLE` |
| **`LIBIDO_BROAD`** | `CRIM SEXUAL ASSAULT`<br>`SEX OFFENSE` | `Rape`<br>`Other Sex Offenses (Not Commercialized)` | `RAPE`<br>`SEX CRIMES` | `RAPE FORCIBLE`<br>`RAPE ATTEMPTED`<br>`BATTERY WITH SEXUAL CONTACT`<br>`ORAL COPULATION`<br>`SEXUAL PENETRATION W/FOREIGN OBJECT` | `SEX OFFENSES FORCIBLE` |
| **`PROSTITUTION_PROXY`** | — | `Prostitution and Commercialized Vice` | `PROSTITUTION & RELATED OFFENSES` | — | `PROSTITUTION` |
| **`NARCOTICS`** | `NARCOTICS` | `Narcotic / Drug Law Violations` | `DANGEROUS DRUGS` | — | `DRUG/NARCOTIC` |
| **`PROPERTY_CRIME`** | `THEFT`<br>`BURGLARY`<br>`CRIMINAL DAMAGE` | `Thefts`<br>`Burglary Residential`<br>`Burglary Non-Residential`<br>`Vandalism/Criminal Mischief`<br>`Theft from Vehicle` | `PETIT LARCENY`<br>`GRAND LARCENY`<br>`BURGLARY`<br>`CRIMINAL MISCHIEF & RELATED OF` | `THEFT PLAIN - PETTY ($950 & UNDER)`<br>`THEFT-GRAND ($950.01 & OVER)EXCPTGUNSFOWLLIVESTKPROD`<br>`SHOPLIFTING - PETTY THEFT ($950 & UNDER)`<br>`BURGLARY`<br>`BURGLARY FROM VEHICLE`<br>`VANDALISM - FELONY ($400 & OVER ALL CHURCH VANDALISMS)`<br>`VANDALISM - MISDEAMEANOR ($399 OR UNDER)` | `LARCENY/THEFT`<br>`BURGLARY`<br>`VANDALISM` |
| **`WEAPONS_STRESS`** | `WEAPONS VIOLATION` | `Weapon Violations` | `DANGEROUS WEAPONS` | `BRANDISH WEAPON` | `WEAPON LAWS` |
| **`DUI_EXTENDED`** | — | `DRIVING UNDER THE INFLUENCE` | `INTOXICATED & IMPAIRED DRIVING` | — | `DRIVING UNDER THE INFLUENCE` |
| **`DISORDERLY_PUBLIC`** | — | `Disorderly Conduct` | — | — | `DISORDERLY CONDUCT` |
| **`PURE_BURGLARY`** | `BURGLARY` | `Burglary Residential` | `BURGLARY` | `BURGLARY` | `BURGLARY` |
| **`PURE_THEFT`** | `THEFT` | `Thefts` | `PETIT LARCENY` | `THEFT PLAIN - PETTY ($950 & UNDER)` | `LARCENY/THEFT` |
| **`PURE_VEHICLE_THEFT`** | `MOTOR VEHICLE THEFT` | `Motor Vehicle Theft` | `GRAND LARCENY OF MOTOR VEHICLE` | `VEHICLE - STOLEN` | `VEHICLE THEFT` |
| **`PURE_ASSAULT`** | `BATTERY` | `Aggravated Assault No Firearm` | `ASSAULT 3 & RELATED OFFENSES` | `BATTERY - SIMPLE ASSAULT` | `ASSAULT` |
| **`PURE_VANDALISM`** | `CRIMINAL DAMAGE` | `Vandalism/Criminal Mischief` | `CRIMINAL MISCHIEF & RELATED OF` | `VANDALISM - FELONY ($400 & OVER ALL CHURCH VANDALISMS)` | `VANDALISM` |
| **`PURE_ROBBERY`** | `ROBBERY` | `Robbery No Firearm` | `ROBBERY` | `ROBBERY` | `ROBBERY` |