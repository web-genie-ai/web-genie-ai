import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field

from webgenie.challenges.challenge_types import (
    ACCURACY_COMPETITION_TYPE,
    QUALITY_COMPETITION_TYPE,
    SEO_COMPETITION_TYPE,
)
from webgenie.tasks.metric_types import (
    ACCURACY_METRIC_NAME, 
    QUALITY_METRIC_NAME,
    SEO_METRIC_NAME,
)
from webgenie.tasks.task import Task
from webgenie.tasks.solution import Solution


class Challenge(BaseModel):
    task: Optional[Task] = Field(default=None, description="The task to be solved")
    solutions: List[Solution] = Field(default=[], description="The solutions to the task")
    competition_type: str = Field(default="", description="The type of competition")
    session_number: int = Field(default=0, description="The session number")

    async def calculate_scores(self) -> dict[str, np.ndarray]:
        pass


class AccuracyChallenge(Challenge):
    competition_type: str = Field(default=ACCURACY_COMPETITION_TYPE, description="The type of competition")

    async def calculate_scores(self) -> dict[str, np.ndarray]:
        scores = await self.task.generator.calculate_scores(self.task, self.solutions)
        aggregated_scores = scores[ACCURACY_METRIC_NAME] * 0.9 + scores[QUALITY_METRIC_NAME] * 0.1
        return aggregated_scores, scores


class SeoChallenge(Challenge):
    competition_type: str = Field(default=SEO_COMPETITION_TYPE, description="The type of competition")

    async def calculate_scores(self) -> dict[str, np.ndarray]:
        scores = await self.task.generator.calculate_scores(self.task, self.solutions)
        accuracy_scores = scores[ACCURACY_METRIC_NAME]
        seo_scores = scores[SEO_METRIC_NAME]
        aggregated_scores = np.where(accuracy_scores > 0.7, seo_scores, 0)
        return aggregated_scores, scores


class QualityChallenge(Challenge):
    competition_type: str = Field(default=QUALITY_COMPETITION_TYPE, description="The type of competition")

    async def calculate_scores(self) -> dict[str, np.ndarray]:
        scores = await self.task.generator.calculate_scores(self.task, self.solutions)
        accuracy_scores = scores[ACCURACY_METRIC_NAME]
        quality_scores = scores[QUALITY_METRIC_NAME]
        aggregated_scores = np.where(accuracy_scores > 0.7, quality_scores, 0)
        return aggregated_scores, scores
