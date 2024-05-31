import {authTypes} from './auth.types';

export type loginStartType = {
  email: string;
  password: string;
};
export type logOutStartType = {
  // email: string;
  // password: string;
};
type userPayload = {user: string};

// login user action
export const loginStart = (userCredentials: loginStartType) => ({
  type: authTypes.LOGIN_START,
  payload: userCredentials,
});
export const logOutStart = () => ({
  type: authTypes.LOGOUT_START,
  // payload: userCredentials,
});

export const loginSuccess = (user: userPayload) => ({
  type: authTypes.LOGIN_SUCCESS,
  payload: user,
});

export const loginFail = (error: {error: any}) => ({
  type: authTypes.LOGIN_FAIL,
  payload: error,
});

// New user actions
export const signUpStart = (userCredentials: any) => ({
  type: authTypes.SIGN_UP_START,
  payload: userCredentials,
});

export const signUpSuccess = ({user}: userPayload) => ({
  type: authTypes.SIGN_UP_SUCCESS,
  payload: {user},
});

export const signUpFail = (error: userPayload) => ({
  type: authTypes.SIGN_UP_FAIL,
  payload: error,
});

// export const signUpStart = userCredentials => ({
//   type: userActionTypes.SIGN_UP_START,
//   payload: userCredentials,
// });

// export const signUpSuccess = ({user, additionalData}) => ({
//   type: userActionTypes.SIGN_UP_SUCCESS,
//   payload: {user, additionalData},
// });
