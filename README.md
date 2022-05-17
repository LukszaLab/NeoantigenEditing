# NeoantigenEditing

***Neoantigen quality predicts immunoediting in survivors of pancreatic cancer***, Nature 2022


Code for computing neoantigen qualities and for 
performing clone composition predictions.

data folder:

data/Patient_data - folder with phylogenies for each of the patients. Top 5 scoring trees are provided for each patient.
Tree clones are annotated with mutations, predicted neoantigens and clone frequencies.

data/epitope_distance_model_parameters.json - cross-reactivity metric

data/fitness_weights.txt - optimized fitness model weights for each recurrent tumor.

data/iedb.fasta - IEDB epitopes used for the analysis in the paper (downloaded from the IEDB on January 2022)

To run the code:

1. Align each patient's neoantigens to IEDB
```
python align_neoantigens_to_IEDB.py
```

2. Compute neoantigen qualities and fitness of all clones
```
python compute_fitness.py
```

3. Predict clone frequencies in recurrent tumors:
```
python predictions_clones.py
```

4. Compute log-likelihood scores - comparison between the fitness model and the model of neutral evolution of tumors.

```
python predictions_aggregated_loglikelihood_scores.py
```

For any questions please contact:

- [Zachary Sethna](mailto:sethnaz@mskcc.org)

- [Marta Luksza](mailto:marta.luksza@mssm.edu)