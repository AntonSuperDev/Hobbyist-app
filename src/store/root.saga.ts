import {all, call} from 'redux-saga/effects';

import {userSagas} from '../services/auth/auth.saga';
import {itemSagas} from '../services/items/item.sagas';

export default function* rootSaga() {
  yield all([call(userSagas), call(itemSagas)]);
}
