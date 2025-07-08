#-------------------------------------------------------------------------------------------------------------
# BIBLIOTECAS ------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
import torch

# ------------------------------------------------------------------------------------------------------------
# METRICS FOR CLASSIFICATION --------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def accuracy(
    pred_classes: torch.Tensor, 
    true_classes: torch.Tensor
) -> int:
    
    # Comparar con las clases verdaderas
    correct = (pred_classes == true_classes).sum()
    
    # Calcular la precisión como tensor
    acc = correct.float() / true_classes.size(0)
    
    return acc.item()

# ------------------------------------------------------------------------------------------------------------
# METRICS FOR BINARY CLASSIFICATION --------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# def confusion_matrix(
#     pred_classes: torch.Tensor, 
#     true_classes: torch.Tensor
# ) -> dict:
    
#     #
#     confusion_matrix = {
#         "True Positive" : None,
#         "True Negative": None,
#         "False Positive": None,
#         "False Negative": None, 
#     }
    
    
    
#     # Obtener el índice de la clase con mayor probabilidad para cada ejemplo
#     predicted_labels = torch.argmax(pred_classes, dim=1)
    
#     # Comparar con las clases verdaderas
#     correct = (predicted_labels == true_classes).sum()
    
#     # Calcular la precisión como tensor
#     acc = correct.float() / true_classes.size(0)
    
#     return acc

# ------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------
# METRICS FOR MULTICLASS CLASSIFICATION ----------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------





# ------------------------------------------------------------------------------------------------------------