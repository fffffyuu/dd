import { create } from 'zustand';
import { PlantResponse } from '../types/plant';

type PlantState = {
  history: PlantResponse[];
  current?: PlantResponse;
  addResult: (result: PlantResponse) => void;
};

export const usePlantStore = create<PlantState>((set) => ({
  history: [],
  current: undefined,
  addResult: (result) =>
    set((state) => ({
      current: result,
      history: [result, ...state.history].slice(0, 50),
    })),
}));
