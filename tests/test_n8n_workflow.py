import json
import unittest
from pathlib import Path


class N8nWorkflowTests(unittest.TestCase):
    def test_importable_workflow_forwards_body_and_bearer_auth(self):
        workflow_path = Path(__file__).parents[1] / "n8n" / "token-tax-proxy.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        nodes = {node["name"]: node for node in workflow["nodes"]}

        self.assertEqual(
            nodes["Receive Chat Request"]["parameters"]["responseMode"],
            "responseNode",
        )
        request = nodes["Call Token Tax API"]["parameters"]
        self.assertEqual(request["jsonBody"], "={{ $json.body }}")
        self.assertEqual(
            request["headerParameters"]["parameters"][0]["value"],
            "={{ $json.headers.authorization }}",
        )
        self.assertEqual(
            workflow["connections"]["Call Token Tax API"]["main"][0][0]["node"],
            "Return Optimized Completion",
        )


if __name__ == "__main__":
    unittest.main()
