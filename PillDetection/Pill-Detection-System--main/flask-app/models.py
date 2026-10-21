from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class MedicalStore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    google_maps_link = db.Column(db.String(200), nullable=False)
    available_medicines = db.Column(db.String(500), nullable=False)

    def __repr__(self):
        return f'<MedicalStore {self.name}>'

