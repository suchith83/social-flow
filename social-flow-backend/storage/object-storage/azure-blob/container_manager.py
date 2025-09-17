"""
Container (bucket equivalent) management utilities for Azure Blob Storage.
"""

from .client import AzureBlobClient


class AzureBlobContainerManager:
    def __init__(self):
        self.client = AzureBlobClient()

    def create_container(self, name: str):
        container_client = self.client.client.create_container(name)
        return container_client

    def delete_container(self, name: str):
        self.client.client.delete_container(name)

    def list_containers(self):
        return [c["name"] for c in self.client.client.list_containers()]

    def list_blobs(self, container_name: str, prefix: str = ""):
        container_client = self.client.container_client(container_name)
        return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]
