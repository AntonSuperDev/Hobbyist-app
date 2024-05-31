import {all, call, put, takeLatest} from 'redux-saga/effects';
import {
  loginFail,
  loginStartType,
  signUpFail,
  signUpSuccess,
} from './auth.action';

import AsyncStorage from '@react-native-async-storage/async-storage';
import {authTypes} from './auth.types';
import axios from 'axios';
import { BASE_URL } from '../../config';

const SIGNUP_URL =
  `${BASE_URL}/api/v1/signup`;
const LOGIN_URL =
  `${BASE_URL}/api/v1/signin`;
// Existing uer login credentials
export function* userLogin({payload: {email, password}}: loginStartType) {
  try {
    const {data} = yield call(axios.post, LOGIN_URL, {
      email,
      password,
    });
    console.log(data.data);
    AsyncStorage.setItem('token', data.data.token);
    yield put(signUpSuccess({user: data.data.token}));
  } catch (error) {
    yield put(loginFail({message: 'Invalid email or password'}));
  }
}

export function* userLogout({} : logOutStartType) {
  // try {
  //   const {data} = yield call(axios.post, LOGIN_URL, {
  //     email,
  //     password,
  //   });
  // console.log(data.data);
  // AsyncStorage.setItem('token', data.data.token);
  AsyncStorage.removeItem('token');
  yield put(signUpSuccess({user: null}));
  // } catch (error) {
  //   yield put(loginFail({message: 'Invalid email or password'}));
  // }
}

// New User Sign Up
export function* userSignUp({
  payload: {fullname, email, password, c_password},
}) {
  try {
    const {data} = yield call(axios.post, SIGNUP_URL, {
      fullname,
      email,
      password,
      c_password,
    });
    console.log(data);
    AsyncStorage.setItem('token', data.data.token);
    yield put(signUpSuccess(data));
  } catch (error: any) {
    console.log('failiar signup');
    yield put(signUpFail({message: 'This Email is Already Registered'}));
  }
}

export function* onLoginStart() {
  yield takeLatest(authTypes.LOGIN_START, userLogin);
}
export function* onLogOutStart() {
  yield takeLatest(authTypes.LOGOUT_START, userLogout);
}

export function* onSignUpStart() {
  yield takeLatest(authTypes.SIGN_UP_START, userSignUp);
}

export function* userSagas() {
  yield all([call(onLoginStart), call(onSignUpStart)]);
}
