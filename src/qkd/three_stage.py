from protocol import StackProtocol


def pair_3stage_protocols(sender: "ThreeStageProtocol", receiver: "ThreeStageProtocol") -> None:
    sender.another = receiver
    receiver.another = sender
    sender.role = 0
    receiver.role = 1


class ThreeStageProtocol(StackProtocol):
    def __init__(self):
        return None