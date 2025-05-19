from jiwer import wer,compute_measures,cer




def compute_wer(reference: str, prediction: str) -> float:
    """
    Computes the Word Error Rate (WER) between prediction and reference text.

    Args:
        reference (str): Ground truth/reference sentence.
        prediction (str): Predicted sentence.
        

    Returns:
        float: Word Error Rate (0 to 1).
    """

    #measures=compute_measures()
    return wer(reference, prediction)




def compute_cer(reference: str, prediction: str) -> float:
    """
    Computes the Character Error Rate (CER) between prediction and reference text.

    Args:
        prediction (str): Predicted sentence.
        reference (str): Ground truth/reference sentence.

    Returns:
        float: Character Error Rate (0 to 1).
    """
    return cer(reference, prediction)




def text_field_accuracy(correct_fields: int, total_fields: int) -> float:
    """
    Computes field-level accuracy percentage.

    Args:
        correct_fields (int): Number of correctly predicted fields.
        total_fields (int): Total number of fields.

    Returns:
        float: Field accuracy (0 to 100).
    """
    if total_fields == 0:
        return 0.0
    return (correct_fields / total_fields) * 100


def document_Level_accuracy(correct_documents: int, total_documents: int) -> float:
    """
    Computes document-level accuracy percentage.

    Args:
        correct_documents (int): Number of documents with all fields correct.
        total_documents (int): Total number of documents.

    Returns:
        float: Document accuracy (0 to 100).
    """
    if total_documents == 0:
        return 0.0
    return (correct_documents / total_documents) * 100


def compute_final_score(wer: float, cer: float, field_acc: float, doc_acc: float) -> float:
    """
    Computes the final weighted score.

    Args:
        wer (float): Word Error Rate (0 to 1).
        cer (float): Character Error Rate (0 to 1).
        field_acc (float): Field accuracy (0 to 100).
        doc_acc (float): Document accuracy (0 to 100).

    Returns:
        float: Final score (0 to 100).
    """
    return (
        0.35 * (100 - wer * 100) +
        0.35 * (100 - cer * 100) +
        0.15 * field_acc +
        0.15 * doc_acc
    )


def compute_efficiency(t_avg: float, m_avg: float) -> float:
    """
    Computes the efficiency score based on average time and memory usage.

    Args:
        t_avg (float): Average processing time per document (in seconds).
        m_avg (float): Average memory usage (in MB or GB).

    Returns:
        float: Efficiency score.
    """
    if t_avg == 0 or m_avg == 0:
        return float('inf')  # Perfect efficiency (not realistic)
    return 1 / (t_avg * m_avg)
