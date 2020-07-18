Published on aavso.org (https://aavso.org)
Home > Format of the Data File
JD: The Julian Date [1] of the observation.
Magnitude: The magnitude estimate of the observation. A < sign means it was a null observation "fainter than" the magnitude given.
Uncertainty: Uncertainty (error) of the observation as submitted by the observer
HQuncertainty: Uncertainty (error) of the observation as calculated by AAVSO Headquarters
Band: Bandpass of the observation. The allowable filters are listed here [2].
Observer Code: This is a unique ID assigned to each observer.
Comment Code: Comment codes submitted by the observer. A list of codes is here [3].
Comp Star 1: The comparison star(s) used to make the visual estimate. If photometric, this is the comparison (C) star ID.
Comp Star 2: The comparison star(s) used to make the visual estimate. If photometric, this is the check (K) star ID.
Charts: The charts used to find the field and locate the comparison stars and their values. As of July, 2008 new charts were issued with a Chart ID format of XXXXY where XXXX is a number and Y can be any combination of letters. You can visit our Variable Star Plotter [4] and type in that Chart ID to see the exact chart the observer used to make that observation. For Chart IDs that are not in that format, contact AAVSO HQ and we can e-mail you a copy of the chart used in the observation.
Comments: Comments on the observation, usually from the observer
Transform: If transformation coefficients were applied to the observation then this will be "Yes".
Airmass: The airmass of the observation.
Validation Flag: This flag describes the level of validation [5] of the observation.
V means the observation has passed our validation tests.
T means that during the validation phase it was flagged discrepant and should be used with extreme caution.
Z means it has only undergone pre-validation, meaning it was checked for typos and data input errors only.
U means it has not been validated at all and should be used with caution.
Cmag: Supplied magnitude of the comparison star
Kmag: Measured magnitude of the check star
HJD: Heliocentric Julian Date
Name: Name of the star
Affiliation: Organization affiliated with the observer that took this data.  See this link [6] for a list of affiliation codes and their corresponding Organizations.
Magnitude Type: This field describes the kind of magnitude used in the observation:
STD - standard magnitude.
DIFF - differential magnitude. The value of Comp Star 1 is needed to compute the standard magnitude.
STEP - un-reduced step magnitude. These magnitudes are given as 0.0 and the step sequence may be found in the Comment Code field.
Group: Denotes what group of filters were used for this observation.
ADS Reference: The article identifier in CDS/ADS Bibliographic Code [7] format
Digitizer: The digitizer code for the individual who digitized and submitted the data (identical to the individual's observer code if they are already an AAVSO observer).
Credit: Organization [8] to be credited for the observation.
Note: The AAVSO database dates back to the 19th century. As a result, it changed over time and fields were added and removed. As a result, some of the fields may be empty in early observations.

Tags:
format [9]
data [10]
data download format [11]
AID [12]
Submitted by willmcmain on Tue, 06/15/2010 - 10:45
Source URL: https://aavso.org/format-data-file
Links
[1] https://aavso.org/jd-calculator
[2] https://aavso.org/filters
[3] https://aavso.org/aavso-visual-file-format
[4] https://aavso.org/vsp
[5] https://aavso.org/aavso-validation-project
[6] https://aavso.org/observer-affiliations-table
[7] http://adsabs.harvard.edu/bib_abs.html
[8] https://aavso.org/database-credit-table
[9] https://aavso.org/category/tags/format
[10] https://aavso.org/category/tags/data
[11] https://aavso.org/tags/data-download-format
[12] https://aavso.org/category/tags/aid

