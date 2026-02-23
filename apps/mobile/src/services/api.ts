import axios from 'axios';
import { PlantResponse } from '../types/plant';

const API_BASE = process.env.EXPO_PUBLIC_API_BASE ?? 'http://localhost:8000/v1';

export async function identifyPlant(uri: string): Promise<PlantResponse> {
  const form = new FormData();
  form.append('image', {
    uri,
    name: 'plant.jpg',
    type: 'image/jpeg',
  } as any);

  const { data } = await axios.post<PlantResponse>(`${API_BASE}/identify`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 10000,
  });

  return data;
}
