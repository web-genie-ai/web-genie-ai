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
    ImageTaskSeoCompetition,
)


RESERVED_WEIGHTS = {
    ImageTaskAccuracyCompetition.COMPETITION_TYPE: 90,
    ImageTaskQualityCompetition.COMPETITION_TYPE: 10,
    ImageTaskSeoCompetition.COMPETITION_TYPE: 20,
}

