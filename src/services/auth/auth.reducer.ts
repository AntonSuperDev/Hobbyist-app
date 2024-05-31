import {AnyAction} from 'redux';
import {authTypes} from './auth.types';

export type userAuth = {
  user: null;
  isLoading: boolean;
  error: null;
};

const initialState: userAuth = {
  user: null,
  isLoading: false,
  error: null,
};

const authReducer = (state = initialState, action: AnyAction) => {
  switch (action.type) {
    case authTypes.LOGIN_START:
    case authTypes.LOGOUT_START:
    case authTypes.SIGN_UP_START:
      return {...state, isLoading: true};

    case authTypes.LOGIN_SUCCESS:
      return {...state, user: action.payload, error: null, isLoading: false};

    case authTypes.SIGN_UP_SUCCESS:
      return {...state, user: action.payload, error: null, isLoading: false};

    case authTypes.LOGIN_FAIL:
    case authTypes.SIGN_UP_FAIL:
      return {...state, error: action.payload, isLoading: false};

    default:
      return state;
  }
};

export default authReducer;
