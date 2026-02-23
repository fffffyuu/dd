import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { PlantResponse } from '../types/plant';

export function ResultCard({ result }: { result: PlantResponse }) {
  return (
    <View style={styles.card}>
      <Text style={styles.title}>{result.selected.common_name}</Text>
      <Text style={styles.subtitle}>{result.selected.scientific_name}</Text>
      <Text style={styles.metric}>Confidence: {(result.selected.confidence * 100).toFixed(1)}%</Text>
      <Text style={styles.body}>{result.description}</Text>
      <Text style={styles.section}>Care Guide</Text>
      <Text style={styles.body}>💧 {result.care.watering}</Text>
      <Text style={styles.body}>☀️ {result.care.sunlight}</Text>
      <Text style={styles.body}>🌱 {result.care.soil}</Text>
      <Text style={styles.body}>🌡️ {result.care.temperature}</Text>
      {result.disease_report && (
        <>
          <Text style={styles.section}>Health Diagnosis</Text>
          <Text style={styles.body}>{result.disease_report.diagnosis}</Text>
          <Text style={styles.body}>{result.disease_report.treatment}</Text>
        </>
      )}
      <Text style={styles.latency}>Inference: {result.processing_ms} ms</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#111827',
    borderRadius: 16,
    padding: 16,
    marginTop: 12,
  },
  title: { fontSize: 24, color: '#F9FAFB', fontWeight: '700' },
  subtitle: { fontSize: 14, color: '#9CA3AF', marginBottom: 8 },
  metric: { color: '#34D399', fontWeight: '600', marginBottom: 8 },
  section: { color: '#C4B5FD', marginTop: 10, fontWeight: '700' },
  body: { color: '#D1D5DB', marginTop: 4 },
  latency: { color: '#6EE7B7', marginTop: 10, fontSize: 12 },
});
