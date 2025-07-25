class Patient:
    def __init__(self, initial_age, initial_joint_score):
        self.alive = True
        self.age = initial_age
        self.jointScore = initial_joint_score
        self.state = "alive_wo_arthropathy"
        self.treatment = "prophylaxis"
