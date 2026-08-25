# NeoantigenEditing

***Neoantigen quality predicts immunoediting in survivors of pancreatic cancer***, Nature 2022


Code for computing neoantigen qualities and for 
performing clone composition predictions.

The input data are the following:

**data/Patient_data** - folder with phylogenies for each of the patients. Top 5 scoring trees are provided for each patient.
Tree clones are annotated with mutations, predicted neoantigens and clone frequencies.

**data/epitope_distance_model_parameters.json** - cross-reactivity metric

**data/fitness_weights.txt** - optimized fitness model weights for each recurrent tumor.

**data/iedb.fasta** - IEDB epitopes used for the analysis in the paper (downloaded from the IEDB on January 2022)

## Requirements

For local execution, install Python 3 and the Python packages imported by the
scripts. The alignment step also requires the NCBI `blastp` executable to be
installed and available on your `PATH`:

```bash
blastp -version
```

Alternatively, use the Docker image described below, which includes Python,
the required packages, and BLAST.

## Running the scripts

Run these commands from the repository root. The first two scripts accept
either one sample file or a patient-data folder. The options are documented in
the parameter guide below.

1. Align each patient's neoantigens to IEDB
```
python3 align_neoantigens_to_IEDB.py \
	--fasta data/iedb.fasta \
	--patient_folder data/Patient_data
```

2. Compute neoantigen qualities and fitness of all clones
```
python3 compute_fitness.py \
	--alignment <alignment_file> \
	--patient_folder data/Patient_data \
	--kd_cutoff_fitness <maximum_kd>
```

3. Predict clone frequencies in recurrent tumors:
```
python3 predictions_clones.py
```

4. Compute log-likelihood scores - comparison between the fitness model and the model of neutral evolution of tumors.

```
python3 predictions_aggregated_loglikelihood_scores.py
```

The alignment command creates `iedb_alignments_<patient>.txt` files in the
current directory. `compute_fitness.py` creates an `_annotated.json` file next
to each input JSON file. The prediction scripts read the annotated files and
write their output to `Results/`.

## Parameter guide

### `align_neoantigens_to_IEDB.py`

| Parameter | Required | Description |
| --- | --- | --- |
| `--fasta` | Yes | Path to the IEDB FASTA file, for example `data/iedb.fasta`. |
| `--sample_file` | No | Process one sample JSON file. Mutually exclusive with `--patient_folder`. |
| `--patient_folder` | No | Process the primary sample files under each patient directory, for example `data/Patient_data`. Mutually exclusive with `--sample_file`. |

One of `--sample_file` or `--patient_folder` must be supplied. The script
invokes `blastp`, so confirm that it is on `PATH` before running.

### `compute_fitness.py`

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `--alignment` | Yes | None | Tab-delimited alignment file produced by the alignment step. |
| `--sample_file` | No | None | Process one sample JSON file. Mutually exclusive with `--patient_folder`. |
| `--patient_folder` | No | None | Process the primary and recurrent sample files under each patient directory. Mutually exclusive with `--sample_file`. |
| `--kd_cutoff_fitness` | Yes | None | Maximum neoantigen Kd value included in the fitness calculation. |
| `--a_param` | No | `22.897590714815188` | Fitness-model weight for `a`. |
| `--k_param` | No | `1` | Fitness-model weight for `k`. |
| `--w_param` | No | `0.22402192838740312` | Weight for the cross-reactivity component versus the affinity component. |

One of `--sample_file` or `--patient_folder` must be supplied. To process a
single sample, replace `--patient_folder data/Patient_data` with, for example:

```bash
python3 compute_fitness.py \
	--alignment iedb_alignments_11-LTS.txt \
	--sample_file data/Patient_data/11-LTS/Primary/11_LTS_primary_tumor.json \
	--kd_cutoff_fitness 500
```

### Prediction scripts

`predictions_clones.py` and
`predictions_aggregated_loglikelihood_scores.py` do not currently define
command-line parameters. Both scripts use the following paths relative to the
current working directory:

| Path | Purpose |
| --- | --- |
| `data/Patient_data` | Primary and recurrent patient JSON files. |
| `data/fitness_weights.txt` | Optimized fitness weights used by clone predictions. |
| `Results/` | Output tables and figures. The directory is created when needed. |

## Docker

The repository includes a `Dockerfile`. A published image is also available:

```bash
docker pull ghcr.io/mskcc-omics-workflows/neoantigen-editing:1.1
```

To use the published image, mount the repository into the container and run
the scripts from the mounted directory:

```bash
IMAGE=ghcr.io/mskcc-omics-workflows/neoantigen-editing:1.1

docker run --rm -v "$PWD":/workspace -w /workspace "$IMAGE" \
	python3 /usr/bin/align_neoantigens_to_IEDB.py \
	--fasta data/iedb.fasta \
	--patient_folder data/Patient_data
```

The same pattern can be used for the other scripts, replacing the script path
and arguments as needed. The image includes `blastp`, so no host BLAST
installation is required when running inside the container.

To build the image locally instead:

```bash
docker build -t neoantigen-editing .
```

For any questions please contact:

- [Zachary Sethna](mailto:sethnaz@mskcc.org)

- [Marta Luksza](mailto:marta.luksza@mssm.edu)

- [Ben Greenbaum](mailto:greenbab@mskcc.org)

- [Vinod Balachandran](mailto:balachav@mskcc.org)