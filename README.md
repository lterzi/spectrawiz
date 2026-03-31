# SpectraWiz

**SpectraWiz** is a modular Python package for interactive radar spectra exploration, forward simulation, and raw radar data processing.  
It is designed for atmospheric radar scientists and engineers, providing both a user-friendly Streamlit web app and powerful command-line and Python APIs.

---

## Features

### 1. **Interactive Explorer (Streamlit App)**
- Visualize radar time-height data, Doppler spectra, and profiles.
- Select variables, time, and range interactively.
- Overlay measured and forward-simulated spectra (rain/snow) for direct comparison.
- Control simulation parameters (rain rate, snow parameters, gamma DSD, turbulence, noise, wind, etc.) with intuitive sliders.
- View and compare simulated Particle Size Distributions (PSD).
- Flexible LUT (lookup table) selection for rain and snow scattering.
- All figures update in real time as you adjust parameters.

### 2. **Forward Spectrum Simulation**
- Simulate Doppler spectra for rain and snow using a generalized gamma drop size distribution (DSD).
- User control over DSD parameters: N₀, λ (lambda), γ (gamma).
- Supports both rain and snow LUTs for scattering properties.
- Adjustable turbulence, wind, noise, and radar beam width.
- Compare simulated and measured spectra side-by-side.

### 3. **Raw Radar Data Processing**
- High-level command-line tool to process raw radar files by day.
- Supports multiple radar backends (Metek, RPG, auto-detection).
- Hourly or file-based processing, with optional time regridding.
- Handles moments, LDR, and polarimetric variables.
- Smart file skipping and overwrite options.
- Debugging output for batch processing.

### 4. **Modular Python Package**
- Clean `src/`-layout package for easy import and extension.
- All core functionality available as Python API for scripting and notebooks.
- Well-structured modules: `explorer`, `radar_simulator`, `process_data`, `backends`, and more.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/spectrawiz.git
cd spectrawiz
pip install -e .
```

---

## Usage

### **1. Interactive Explorer**

Launch the Streamlit app:

```bash
streamlit run src/spectrawiz/explorer.py
```

Or use a launcher script if provided.

### **2. Command-Line Data Processing**

Process a day of radar data:

```bash
spectrawiz-process --date 20250910 --raw_path /data/raw --output_dir /data/processed --hourly --include_moments --debugging
```

See all options with:

```bash
spectrawiz-process --help
```

### **3. Python API**

Use in your own scripts or notebooks:

```python
from spectrawiz.process_data import process_day
written = process_day(date="20250910", raw_path="...", output_dir="...")
```

---

## Project Structure

```
spectrawiz/
├── pyproject.toml
├── README.md
├── src/
│   └── spectrawiz/
│       ├── __init__.py
│       ├── explorer.py
│       ├── radar_simulator.py
│       ├── process_data.py
│       ├── processing_common.py
│       └── backends/
│           ├── __init__.py
│           ├── base.py
│           ├── metek.py
│           ├── rpg.py
│           └── registry.py
└── tests/
    └── test_simulator.py
```

---

## Adding a New Processing Backend

To add support for another radar backend (e.g., MRR):

1. **Create a new backend module:**
   - Add a new Python file in `src/spectrawiz/backends/`, e.g. `mrr.py`.
   - Implement a class (e.g., `MRRBackend`) for reading and processing MRR files, following the structure of existing backends like `metek.py` or `rpg.py`.

2. **Expose your backend:**
   - In `src/spectrawiz/backends/__init__.py`, import your backend class and add it to `__all__`:
     ```python
     from .mrr import MRRBackend
     __all__ = [
         ...,
         "MRRBackend",
     ]
     ```

3. **Register your backend:**
   - In `src/spectrawiz/process_data.py`, import your backend and register it:
     ```python
     from .backends import MRRBackend, register_backend
     register_backend(MRRBackend(), overwrite=True)
     ```

4. **Implement required interface:**
   - Your backend should implement the methods required by the processing framework (see `MetekBackend` or `RPGBackend` for examples).

5. **Test your backend:**
   - Add tests in the `tests/` directory to ensure your backend works as expected.

6. **Use your new backend:**
   - Specify `--radar_type mrr` when running the processing CLI, or set it in your scripts.

**Note:**  
You do not need to modify `registry.py` directly.  
Just expose your backend in `__init__.py` and register it in `process_data.py`.

---

## Backend Auto-Detection

SpectraWiz supports **automatic backend selection** based on the contents of your radar files.  
This is achieved by each backend implementing a `can_handle()` static method, which inspects the input data and returns `True` if the backend can process it.

### How Auto-Detection Works

- When processing a file with `--radar_type auto` (the default), SpectraWiz will:
  1. Load the file as an `xarray.Dataset`.
  2. Iterate through all registered backends.
  3. Call each backend’s `can_handle(ds)` method.
  4. Use the first backend that returns `True`.

### How to Make Your Backend Auto-Detectable

To enable auto-detection for your new backend, **implement a `can_handle()` static method** in your backend class.  
This method should inspect the dataset and return `True` if it matches your backend’s expected format.

#### Example from `rpg.py`:

```python
class RPGBackend(RadarBackend):
    name = "rpg"

    @staticmethod
    def can_handle(ds: xr.Dataset) -> bool:
        """
        Auto-detection rule for RPG input.

        Uses presence of `C1Range` as a signature variable.
        """
        return "C1Range" in ds.variables or "C1Range" in ds.coords
```

#### For Your New Backend

- Choose a variable or attribute unique to your file format.
- Implement `can_handle()` to check for that signature.

**Example for an MRR backend:**

```python
class MRRBackend(RadarBackend):
    name = "mrr"

    @staticmethod
    def can_handle(ds: xr.Dataset) -> bool:
        # Suppose MRR files always have a variable called 'MRR_Spec'
        return "MRR_Spec" in ds.variables
```

### Usage

- When you run the processing CLI with `--radar_type auto` (or omit the flag), your backend will be automatically selected if its `can_handle()` method returns `True` for the input file.

**Tip:**  
Refer to `rpg.py` and other backend modules for examples of robust auto-detection logic.  
Make sure your backend is registered and imported as described in the previous section.

---

## Requirements

- Python 3.8+
- streamlit
- xarray
- numpy
- matplotlib
- pandas

(See `pyproject.toml` for full list.)

---

## Contributing

Contributions, bug reports, and feature requests are welcome!  
Please open an issue or submit a pull request.

---

## License

[MIT License](LICENSE)

---

## Acknowledgements

Developed by L. Terzi and collaborators.  
Inspired by the needs of the atmospheric radar community.

---