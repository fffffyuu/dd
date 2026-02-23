import React, { useState } from 'react';
import { Alert, Button, SafeAreaView, ScrollView, StyleSheet, Text, View, Image } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { identifyPlant } from '../services/api';
import { usePlantStore } from '../state/usePlantStore';
import { ResultCard } from '../components/ResultCard';

export function HomeScreen() {
  const [photoUri, setPhotoUri] = useState<string>();
  const [loading, setLoading] = useState(false);
  const { current, addResult } = usePlantStore();

  const pickImage = async (fromCamera: boolean) => {
    const result = fromCamera
      ? await ImagePicker.launchCameraAsync({ quality: 0.8 })
      : await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });

    if (!result.canceled) {
      setPhotoUri(result.assets[0].uri);
    }
  };

  const runIdentify = async () => {
    if (!photoUri) {
      Alert.alert('Upload a plant image first.');
      return;
    }
    setLoading(true);
    try {
      const response = await identifyPlant(photoUri);
      addResult(response);
    } catch {
      Alert.alert('Identification failed', 'Make sure API is reachable or switch to offline model build.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.root}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.headline}>OpenPlant ID</Text>
        <Text style={styles.sub}>Camera-first AI plant identification for mobile + web.</Text>

        <View style={styles.actions}>
          <Button title="Take Photo" onPress={() => pickImage(true)} />
          <Button title="Pick from Gallery" onPress={() => pickImage(false)} />
        </View>

        <Button title={loading ? 'Identifying...' : 'Identify Plant'} onPress={runIdentify} disabled={loading} />

        {photoUri && <Image source={{ uri: photoUri }} style={styles.preview} />}
        {current && <ResultCard result={current} />}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#030712' },
  container: { padding: 16, paddingBottom: 40 },
  headline: { color: '#F3F4F6', fontSize: 32, fontWeight: '800' },
  sub: { color: '#9CA3AF', marginVertical: 10 },
  actions: { flexDirection: 'row', justifyContent: 'space-between', gap: 12, marginBottom: 12 },
  preview: { width: '100%', height: 260, borderRadius: 16, marginTop: 12 },
});
