import os
from pathlib import Path

import openreview
import pandas as pd

from src.AINSTEIN import logger
from src.AINSTEIN.entity.config_entity import DataIngestionConfig
from src.AINSTEIN.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _extract_value(self, field):
        if isinstance(field, dict):
            return field.get("value", "")
        return field or ""

    def _get_public_submissions(self, client):
        invitations_to_try = [
            f"{self.config.venue_id}/-/Blind_Submission",
            f"{self.config.venue_id}/-/Submission",
        ]

        for invitation in invitations_to_try:
            try:
                logger.info(f"Trying invitation: {invitation}")
                submissions = client.get_all_notes(invitation=invitation)
                if submissions:
                    logger.info(f"Retrieved {len(submissions)} submissions via: {invitation}")
                    return submissions
            except Exception as exc:
                logger.warning(f"Failed for invitation {invitation}: {exc}")

        raise ValueError(
            f"No public submissions found for venue_id={self.config.venue_id}. "
            f"Tried: {invitations_to_try}"
        )
    def download_file(self):
        if os.path.exists(self.config.local_data_file):
            logger.info(f"Data already present at: {self.config.local_data_file}, size: {get_size(Path(self.config.local_data_file))}")
            return Path(self.config.local_data_file)

        logger.info("Starting downloading dataset")

        client = openreview.api.OpenReviewClient(
            baseurl=self.config.source_URL
        )

        submissions = self._get_public_submissions(client)
        rows = []

        for paper in submissions:
            paper_id = getattr(paper, "id", "")
            content = getattr(paper, "content", {}) or {}

            title = self._extract_value(content.get("title", ""))
            abstract = self._extract_value(content.get("abstract", ""))
            pdf_path = self._extract_value(content.get("pdf", ""))
            venue = self._extract_value(content.get("venue", ""))

            if not pdf_path:
                logger.warning(f"Empty pdf_path for paper_id={paper_id}, skipping.")
                continue

            if pdf_path.startswith("/"):
                pdf_url = f"https://openreview.net{pdf_path}"
            else:
                pdf_url = pdf_path

            tier = ""
            venue_lower = venue.lower()

            if "oral" in venue_lower:
                tier = "Oral"
            elif "spotlight" in venue_lower:
                tier = "Spotlight"
            elif "poster" in venue_lower:
                tier = "Poster"
            else:
                logger.warning(f"Unrecognized tier for paper_id={paper_id}, venue='{venue}', skipping.")
                continue

            rows.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                    "tier": tier,
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            raise ValueError("No papers found after filtering tiers. Check venue_id or invitation string.")

        oral_df = df[df["tier"] == "Oral"]
        spotlight_df = df[df["tier"] == "Spotlight"]
        poster_df = df[df["tier"] == "Poster"]

        logger.info(f"Total Oral papers found: {len(oral_df)}")
        logger.info(f"Total Spotlight papers found: {len(spotlight_df)}")
        logger.info(f"Total Poster papers found: {len(poster_df)}")

        oral_sample = oral_df.sample(n=min(213, len(oral_df)), random_state=42)
        spotlight_sample = spotlight_df.sample(n=min(379, len(spotlight_df)), random_state=42)
        poster_sample = poster_df.sample(n=min(622, len(poster_df)), random_state=42)

        curated_dataset = pd.concat(
            [oral_sample, spotlight_sample, poster_sample],
            ignore_index=True
        )

        output_dir = os.path.dirname(self.config.local_data_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        curated_dataset.to_csv(self.config.local_data_file, index=False)

        logger.info(f"Dataset saved at: {self.config.local_data_file}")
        logger.info(f"Dataset size: {get_size(Path(self.config.local_data_file))}")
        logger.info(f"Final curated dataset shape: {curated_dataset.shape}")

        return Path(self.config.local_data_file)