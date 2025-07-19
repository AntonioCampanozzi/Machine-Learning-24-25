# GEM-D: Graph Embedding approach on MOOC's early Dropout prediction

Since the COVID-19 pandemic, Massive Open Online Courses (MOOCs) have gained significant popularity. However, due to their fully online nature, they suffer from a notoriously high dropout rate. Numerous Machine Learning and Deep Learning approaches have been proposed to address this issue. Among the most critical challenges is the lack of earliness in the prediction, needing the gather of substantial amount of information before being able decide whether the dropout is likely or not. 

I present GEM-D, a novel approach to this problem by leveraging graph embeddings to generate synthetic yet plausible user-course interactions. These enriched representations are then used to predict student dropout at an early stage, potentially providing actionable insights into learner engagement.

I investigated the effectiveness of a link prediction approach based on graph embeddings, leveraging the structural properties of the graph constructed from existing interactions between users and courses.

Link prediction is employed as a tool to generate feature vectors representing actions between a new user-course relationship. These vectors are then input to a binary classifier that predicts whether a user will drop out of a course based on their observed actions.

## Repository Structure

In this repository you will find:

- **`datasets/`**  
  Contains the dataset used for training both the embedding models and the classifier.

- **`notebooks/`**  
  Jupyter notebooks for data preprocessing, hyperparameter tuning, training, and validation of all models.

- **`optuna_studies/`**  
  Stores the results of all hyperparameter optimization processes performed using Optuna.

- **`results/`**  
  Contains evaluation results of the models, along with the `entity_to_id` and `relation_to_id` mappings extracted from the triples.

- **`Machine_learning_project_report.pdf`**  
  The full report of the project.
  


## Start the system
- clone the repository:

    ```
    git clone https://github.com/AntonioCampanozzi/ICON_23-24.git
    ```
- go into the directory:
    
    ```
    cd ICON_23-24
    ```
- create virtual environment:

    ```
    python -m venv ML-24-25
    ```
- activate it:
    ```
    ML-24-25/Scripts/activate
    ```

- Install dependencies:

    ```
    pip install -r requirements.txt
    ```

## Use the system

start the main:

```
python main.py
```





   
