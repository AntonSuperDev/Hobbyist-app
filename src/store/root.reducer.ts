import {combineReducers} from '@reduxjs/toolkit';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {persistReducer} from 'redux-persist';
import authReducer from '../services/auth/auth.reducer';
import itemReducer from '../services/items/item.reducer';
//import {RootState} from './store';
//import {userReducer} from './user.reducer';
// export type extendPersistConfig = PersistConfig<RootState> & {
//   whitelist: (keyof RootState)[];
// };
const persistConfig = {
  key: 'root',
  storage: AsyncStorage,
  whitelist: ['user.ss'],
};

const rootReducer = combineReducers({
  user: authReducer,
  items: itemReducer,
});

export default persistReducer(persistConfig, rootReducer);
