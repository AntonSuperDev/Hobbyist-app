export enum userType {
  SET_CURRENT_USER = 'SET_CURRENT_USER',
}
import {AnyAction} from 'redux';
type initialState = {currentUser: null; isLoading: boolean};

const INTIALSTATE: initialState = {
  currentUser: null,
  isLoading: false,
};

export const userReducer = (state = INTIALSTATE, action: AnyAction) => {
  const {type, payload} = action;

  switch (type) {
    case userType.SET_CURRENT_USER:
      return {...state, currentUser: payload};
    default:
      return state;
  }
};
