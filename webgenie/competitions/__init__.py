from webgenie.competitions.competition import Competition
from webgenie.competitions.text_task_competition import TextTaskAccuracyCompetition, TextTaskQualityCompetition
from webgenie.competitions.image_task_competition import ImageTaskAccuracyCompetition, ImageTaskQualityCompetition

RESERVED_WEIGHTS = {
    TextTaskAccuracyCompetition.name: 50,
    TextTaskQualityCompetition.name: 20,
    ImageTaskAccuracyCompetition.name: 90,
    ImageTaskQualityCompetition.name: 10
}



