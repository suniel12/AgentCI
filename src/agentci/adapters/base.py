"""
Base Adapter class.
"""

class BaseAdapter:
    def run(self, agent, input_data):
        raise NotImplementedError
