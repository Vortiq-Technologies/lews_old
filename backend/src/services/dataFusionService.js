/**
 * Data Fusion Service
 * Combines sensor data with satellite data for dual-source ML predictions
 */

import satelliteMlService from './satelliteMlService.js';

class DataFusionService {
  /**
   * Fuse sensor data with satellite data into unified dataset
   * Returns 13 features: 5 sensor + 4 terrain + 4 climate
   */
  async fuseSensorWithSatellite(sensorData) {
    try {
      // Extract location from sensor data or use default (Chandigarh, India)
      const lat = sensorData.latitude || 31.2548;
      const lon = sensorData.longitude || 75.7057;

      console.log(`🔄 Fusing data for location: ${lat}, ${lon}`);

      // Get satellite data for sensor location
      const satelliteData = await satelliteMlService.getSatelliteData(lat, lon);

      // Combine into unified 13-feature dataset
      const fusedData = {
        // ========== SENSOR FEATURES (5) ==========
        soilMoisture: sensorData.soilMoisture || 0,
        waterLevel: sensorData.waterLevel || 0,
        tilt: sensorData.tilt || 0,
        vibration: sensorData.vibration || 0,
        ultrasonicDistance: sensorData.ultrasonicDistance || 0,
        
        // ========== TERRAIN FEATURES (4) ==========
        elevation: satelliteData?.terrain?.elevation || 350,
        slope: satelliteData?.terrain?.slope || 5,
        aspect: satelliteData?.terrain?.aspect || 180,
        rainfall: satelliteData?.rainfall?.last24h || 0,
        
        // ========== CLIMATE FEATURES (4) ==========
        temperature: satelliteData?.climate?.temperature || 25,
        humidity: satelliteData?.climate?.humidity || 60,
        windSpeed: satelliteData?.climate?.windSpeed || 5,
        pressure: 1013,  // Standard atmospheric pressure
        
        // ========== METADATA ==========
        sensorTimestamp: sensorData.timestamp || new Date().toISOString(),
        satelliteTimestamp: satelliteData?.timestamp || new Date().toISOString(),
        location: { lat, lon }
      };

      // Calculate data quality metrics
      const sensorAge = Date.now() - new Date(fusedData.sensorTimestamp).getTime();
      const satelliteAge = Date.now() - new Date(fusedData.satelliteTimestamp).getTime();

      fusedData.dataQuality = {
        sensorStatus: sensorAge < 60000 ? 'LIVE' : sensorAge < 300000 ? 'RECENT' : 'STALE',
        satelliteStatus: satelliteAge < 3600000 ? 'FRESH' : 'CACHED',
        sensorAge: this.formatAge(sensorAge),
        satelliteAge: this.formatAge(satelliteAge),
        completeness: this.calculateCompleteness(fusedData)
      };

      console.log(`✅ Data fusion complete: ${fusedData.dataQuality.completeness}% complete`);

      return fusedData;
    } catch (error) {
      console.error('❌ Data fusion error:', error.message);
      
      // Return sensor data only as fallback
      return {
        soilMoisture: sensorData.soilMoisture || 0,
        waterLevel: sensorData.waterLevel || 0,
        tilt: sensorData.tilt || 0,
        vibration: sensorData.vibration || 0,
        ultrasonicDistance: sensorData.ultrasonicDistance || 0,
        elevation: 350,
        slope: 5,
        aspect: 180,
        rainfall: 0,
        temperature: 25,
        humidity: 60,
        windSpeed: 5,
        pressure: 1013,
        dataQuality: {
          sensorStatus: 'ACTIVE',
          satelliteStatus: 'UNAVAILABLE',
          completeness: 50,
          error: error.message
        }
      };
    }
  }

  /**
   * Format age in human-readable format
   */
  formatAge(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return `${seconds}s ago`;
  }

  /**
   * Calculate data completeness percentage
   */
  calculateCompleteness(data) {
    const requiredFields = [
      'soilMoisture', 'waterLevel', 'tilt', 'vibration', 'ultrasonicDistance',
      'elevation', 'slope', 'aspect', 'rainfall',
      'temperature', 'humidity', 'windSpeed', 'pressure'
    ];

    const nonZeroFields = requiredFields.filter(field => 
      data[field] !== undefined && data[field] !== null && data[field] !== 0
    );

    return Math.round((nonZeroFields.length / requiredFields.length) * 100);
  }

  /**
   * Validate fused data before sending to ML API
   */
  validateFusedData(fusedData) {
    const requiredFeatures = [
      'soilMoisture', 'waterLevel', 'tilt', 'vibration', 'ultrasonicDistance',
      'elevation', 'slope', 'aspect', 'rainfall',
      'temperature', 'humidity', 'windSpeed', 'pressure'
    ];

    const missingFeatures = requiredFeatures.filter(feature => 
      fusedData[feature] === undefined || fusedData[feature] === null
    );

    if (missingFeatures.length > 0) {
      console.warn(`⚠️  Missing features: ${missingFeatures.join(', ')}`);
      return false;
    }

    return true;
  }
}

export default new DataFusionService();
