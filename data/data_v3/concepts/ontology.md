# Chinese Traffic Sign Physical Concept Ontology (data_v3)

Grounded in **National Standard of P.R. China GB 5768-2009 (Road Traffic Signs and Markings)** and the **1968 UNECE Vienna Convention on Road Signs and Signals**:

## 1. Shape (4 dimensions)
- `circle`: Regulatory prohibitory (禁令标志) and mandatory/directional (指示标志) signs.
- `triangle_up`: Warning / hazard alert signs (警告标志).
- `triangle_down`: Slow down / Give way (减速慢行 - Class 36).
- `rectangle`: Auxiliary and directional indication signs.

## 2. Colour Profile (5 dimensions)
- `white_red`: White background with red circular outer border and red diagonal slash (Prohibition).
- `blue_white`: Blue background with white arrows / pictograms (Mandatory directions).
- `yellow_white`: Yellow background with black border / black pictograms (Hazard warnings).
- `white_black`: White background with black cancellation slashes (End of prohibition / speed limit).
- `red_white`: Red background with white elements.

## 3. Border Type (3 dimensions)
- `red_border`: Standard prohibition circular rim.
- `white_border`: Inner rim for mandatory blue discs.
- `black_border`: Triangular warning outer border and cancellation slashes.

## 4. Symbol Category (5 dimensions)
- `speed_numeral`: Speed restrictions (Classes 0-7: 5 to 80 km/h).
- `prohibition_restriction`: Turning, overtaking, entry, honking, and parking prohibitions (Classes 8-14, 16-18, 55-57).
- `mandatory_direction`: Compulsory maneuvers, lane usages, and roundabout (Classes 21-34).
- `warning_hazard`: Road geometries, pedestrian/school crossings, railway intersections, and hazards (Classes 35, 37-54).
- `priority_geometry`: Right-of-way regulations (Class 36 Slow down / Yield).
- `deregulation_cancel`: End of restrictions (Classes 15, 19, 20).
