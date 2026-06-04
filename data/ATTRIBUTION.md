# Dataset Attribution & License

This project uses the **Hotel Booking Demand** dataset. The raw data is **not committed** to this
repository (see `data/raw/DATASET_MANIFEST.yml` and the project README for how to obtain it).

## Source & authorship
- **Dataset:** Hotel Booking Demand (course release v1)
- **Original authors:** Nuno António, Ana de Almeida, Luís Nunes
- **Peer-reviewed reference:** António, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.*
  **Data in Brief, 22, 41–49.** https://doi.org/10.1016/j.dib.2018.11.126
- **Kaggle distribution:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- **Course release fingerprint (SHA-256):** `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`
  (`hotel_bookings_course_release_v1.csv`, 119,390 rows × 32 columns)

## License
Distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license, as stated on
the Kaggle dataset page. Full license text: https://creativecommons.org/licenses/by/4.0/

Under CC BY 4.0 we may share and adapt the data for any purpose, including commercially, **provided we give
appropriate credit**, link to the license, and indicate if changes were made. This project:
- credits the original authors (above and in the report references);
- links to the license (above);
- documents every transformation applied to the data in `src/preprocessing/` and the report Methods section
  (no changes are made to the raw source files — all transforms are applied in-pipeline at load time).

## Citation (BibTeX)
```bibtex
@article{antonio2019hotel,
  title   = {Hotel booking demand datasets},
  author  = {Ant{\'o}nio, Nuno and de Almeida, Ana and Nunes, Lu{\'i}s},
  journal = {Data in Brief},
  volume  = {22},
  pages   = {41--49},
  year    = {2019},
  doi     = {10.1016/j.dib.2018.11.126}
}
```

## Note for the report
The Discussion/Dataset-documentation section must cite the reference above and state the CC BY 4.0 terms.
This file is the machine-and-human-readable attribution asset backing that requirement (brief: *"the report must
include the dataset's usage terms/license … and any relevant restrictions"*).
