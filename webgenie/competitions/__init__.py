from webgenie.competitions.text_task_competition import TextTaskCompetition, TextTaskAccuracyCompetition, TextTaskQualityCompetition
from webgenie.competitions.image_task_competition import ImageTaskCompetition, ImageTaskAccuracyCompetition, ImageTaskQualityCompetition

RESERVED_WEIGHTS = {
    TextTaskAccuracyCompetition: 50,
    TextTaskQualityCompetition: 20,
    ImageTaskAccuracyCompetition: 90,
    ImageTaskQualityCompetition: 10
}



