from webgenie.competitions.competition import (
    Competition, 
    ACCURACY_METRIC_NAME, 
    QUALITY_METRIC_NAME,
)
from webgenie.competitions.text_task_competition import (
    TextTaskCompetition,
    TextTaskAccuracyCompetition, 
    TextTaskQualityCompetition,
)
from webgenie.competitions.image_task_competition import (
    ImageTaskCompetition,
    ImageTaskAccuracyCompetition, 
    ImageTaskQualityCompetition,
)


RESERVED_WEIGHTS = {
    TextTaskAccuracyCompetition.COMPETITION_TYPE: 50,
    TextTaskQualityCompetition.COMPETITION_TYPE: 20,
    ImageTaskAccuracyCompetition.COMPETITION_TYPE: 90,
    ImageTaskQualityCompetition.COMPETITION_TYPE: 10
}

