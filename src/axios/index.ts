import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { BASE_URL } from '../config';

axios.defaults.baseURL = `${BASE_URL}/`;

const apiInstance = axios.create();

apiInstance.interceptors.request.use(
  async config => {
    const token = await AsyncStorage.getItem('token');
    // console.log('token', token);
    if (token) {
      config.headers.Authorization = 'Bearer ' + token;
    }
    return config;
  },
  error => {
    return Promise.reject(error);
  },
);

export default apiInstance;
