import AsyncStorage from '@react-native-async-storage/async-storage';

export const STORAGE_KEYS = {
  IS_FULL_APP_PURCHASED: '@is_full_app_purchased',
};

export const storeValue = async (value: number) => {
  try {
    const stringValue = value.toString();
    await AsyncStorage.setItem('@tier_purchase', stringValue);
  } catch (e) {
    console.log(e);
  }
};
// getItem returns a promise that either resolves to stored value when data is found for given key, or returns null otherwise.
export const getValue = async () => {
  try {
    const value = await AsyncStorage.getItem('@tier_purchase');
    if (value === undefined) {
      return null;
    }
    return Number(value);
  } catch (e) {
    console.log(e);
    return null;
  }
};
