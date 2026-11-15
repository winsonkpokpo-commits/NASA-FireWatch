# Raw NASA FIRMS downloads for Benin

These are the original NASA FIRMS ZIP archives for the Benin area, downloaded on 15 November 2026. The archives are kept unmodified. Each contains a standard/archive CSV, an NRT CSV, and FIRMS' readme pointer. Counts and date spans below describe rows present in each CSV; a date span does not imply a detection on every day.

| File | Product | Request ID | Archive rows and dates | NRT rows and dates | ZIP size |
|---|---|---:|---|---|---:|
| `DL_FIRE_M-C61_850991.zip` | MODIS Collection 6.1 (Terra/Aqua) | 850991 | 431,121; 2000-11-01 to 2026-08-25 | 362; 2026-10-16 to 2026-11-15 | 7,221,911 bytes |
| `DL_FIRE_J1V-C2_850992.zip` | VIIRS 375 m, NOAA-20 | 850992 | 512,608; 2018-04-01 to 2026-07-25 | 2,861; 2026-08-04 to 2026-11-15 | 9,104,696 bytes |
| `DL_FIRE_J2V-C2_850993.zip` | VIIRS 375 m, NOAA-21 | 850993 | 161,434; 2023-03-01 to 2026-07-26 | 2,558; 2026-08-04 to 2026-11-15 | 2,925,512 bytes |

The MODIS archive provides the long record from November 2000; VIIRS records begin when each satellite's product becomes available. NASA notes that NRT records are replaced by standard-quality data as they become available (typically after several months). Keep source/product identity and processing version when combining records, and avoid counting overlapping detections twice.

## Source, attribution, and disclaimer

Source: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/), part of NASA LANCE/ESDIS. NASA promotes open sharing of data and requests that third-party redistribution follow its citation guidance and replicate or link to the disclaimer.

Suggested acknowledgement: “We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS), part of NASA's Land, Atmosphere Near real-time Capability for Earth observations (LANCE) and NASA's Earth Science Data and Information System (ESDIS).”

- [NASA FIRMS citation and disclaimer guidance](https://www.earthdata.nasa.gov/earth-observation-data/near-real-time/citation#ed-firms-citation)
- [FIRMS download readme](https://firms.modaps.eosdis.nasa.gov/download/Readme.txt)

FIRMS data are provided “as is.” The disclaimer cautions that their spatial resolution and other characteristics make them unsuitable for tactical decision-making or informing about conditions at a local scale. The archive readmes do not identify a separate license for these downloaded files. This project is independent and is not endorsed by NASA.
