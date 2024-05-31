/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * Generated with the TypeScript template
 * https://github.com/react-native-community/react-native-template-typescript
 *
 * @format
 */
import React, {FC, useState, useEffect} from 'react';
import {StatusBar, LogBox} from 'react-native';
import {ThemeProvider} from 'styled-components/native';
import {theme} from './src/infrastructure/theme';
import {SafeAreaProvider} from 'react-native-safe-area-context';

import {Provider} from 'react-redux';
import {PersistGate} from 'redux-persist/integration/react';
import {store, persistor} from './src/store/store';
LogBox.ignoreLogs([
  "[react-native-gesture-handler] Seems like you're using an old API with gesture components, check out new Gestures system!",
]);

import SplashScreen from './src/components/loading/splash.screen';
import Navigation from './src/infrastructure/navigation';
//import AsyncStorage from '@react-native-async-storage/async-storage';
//import {AuthContext} from './src/services/auth/auth.context';

const App: FC = () => {
  const [isLoading, setIsLoading] = useState<any>(true);

  // const [token, setToken] = useState(null);

  // useEffect(() => {
  //   fetchUser();
  // }, []);
  // const fetchUser = async () => {
  //   setIsLoading(true);
  //   const userData = await AsyncStorage.getItem('token');
  //   if (userData !== null) {
  //     setToken(userData);
  //   }
  //   setIsLoading(false);
  // };

  useEffect(() => {
    setTimeout(() => {
      setIsLoading(false);
    }, 3000);
  }, []);

  return (
    <Provider store={store}>
      <SafeAreaProvider>
        {/* <AuthContext.Provider value={authContext}> */}
        <PersistGate persistor={persistor}>
          <ThemeProvider theme={theme}>
            {/* <HomeScreen /> */}
            {isLoading ? <SplashScreen /> : <Navigation />}
            <StatusBar />
          </ThemeProvider>
        </PersistGate>
        {/* </AuthContext.Provider> */}
      </SafeAreaProvider>
    </Provider>
  );
};

export default App;
