from dataclasses import dataclass


@dataclass
class BaseHandler:

    def run(self):
        return NotImplementedError(
            "This method should be implemented in the child class"
        )
