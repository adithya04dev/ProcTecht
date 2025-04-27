export interface ReceivedMessageData {
  frame: string; // Base64 encoded frame
  detections: {
    fight: FightClassification;
    anomalies: AnomalyClassification;
    objects: ObjectDetection[];
    weapons: WeaponDetection[];
    accidents: AccidentDetection[];
    fire: FireDetection[];
    crack: CrackDetection[];
    tamper: Tamper | null;
    face: FaceDetection[];
    climber: ClimberDetection[];
  };
}
export interface ClimberDetection extends ObjectDetection {
  label: "climber";
}


export interface ObjectDetection {
  label: string;
  confidence: number;
  bbox: number[];
  bbox_std: number[];
  orig_shape: number[];
}
export interface Tamper {
  tamper: boolean;
}
export interface WeaponDetection extends ObjectDetection {
  label:  "knife" | "Pistol"| "gun";
}
export interface FireDetection extends ObjectDetection {
  label: "fire";
}
export interface CrackDetection extends ObjectDetection {
  label: "severe_crack" | "normal_crack";
}
export interface AccidentDetection extends ObjectDetection {
  label: "Accident" | "car_car_accident";
}
export interface FaceDetection extends ObjectDetection {}
export interface Classification {
  predicted_class: number;
  prediction_confidence: number;
  prediction_label: string;
}
export interface FightClassification extends Classification {
  prediction_label: "fight" | "no-fight";
}
export interface AnomalyClassification {
  climbing: boolean;
  violence: boolean;
  suspicious: boolean;
  prediction: {
    predicted_class: number;
    prediction_confidence: number;
    prediction_label: string;
  };
}
