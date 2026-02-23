export type Prediction = {
  scientific_name: string;
  common_name: string;
  confidence: number;
};

export type PlantResponse = {
  top_predictions: Prediction[];
  selected: Prediction;
  description: string;
  care: {
    watering: string;
    sunlight: string;
    soil: string;
    temperature: string;
  };
  disease_report?: {
    likely_disease: string;
    confidence: number;
    diagnosis: string;
    treatment: string;
  };
  processing_ms: number;
};
