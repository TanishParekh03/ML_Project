# Belgian Traffic Sign Physical Concept Ontology (data_v2)

Grounded in the **1968 UNECE Vienna Convention on Road Signs and Signals** and the **Belgian Highway Code (Code de la route / Wegcode)**:

## 1. Shape (6 dimensions)
- `circle`: Regulatory prohibitory (Series C) and mandatory (Series D) signs.
- `triangle_up`: Warning/danger signs (Series A).
- `triangle_down`: Yield / Give way sign (B1).
- `octagon`: Stop sign (B5).
- `diamond`: Priority road signs (B9, B11).
- `rectangle`: Informational / Indication (Series F) and Parking (Series E) signs.

## 2. Colour Profile (5 dimensions)
- `white_red`: White background with red border / red accents.
- `blue_white`: Blue background with white pictograms / white border.
- `yellow_white`: Yellow diamond with white border (Priority road B9).
- `white_black`: White background with black diagonal cancellation (B11).
- `red_white`: Red background with white elements (B5 Stop, C1 No Entry).

## 3. Border Type (4 dimensions)
- `red_border`: Standard warning and prohibitory outer band.
- `white_border`: Inner border for mandatory blue discs, Stop sign, and Priority road.
- `black_border`: Diagonal cancellation stripe.
- `none`: Borderless signs.

## 4. Symbol Category (7 dimensions)
- `warning_hazard`: Hazard warnings (curves, pedestrians, children, road works).
- `priority_geometry`: Right-of-way regulations (Yield, Stop, Priority Road).
- `prohibition_restriction`: Restrictions (No entry, No passing, No trucks, weight limits).
- `speed_numeral`: Speed limits (C43).
- `mandatory_direction`: Obligatory maneuvers (turn directions, roundabout).
- `deregulation_cancel`: End of restriction / End of residential area (B11, F12b).
- `none`: General informational and parking signs.
