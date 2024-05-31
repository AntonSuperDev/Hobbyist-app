import {
  addItemFail,
  addItemSucess,
  getItemListSucess,
  removeItemSucess,
} from './item.action';
import {all, call, put, takeLatest} from 'redux-saga/effects';

import {ItemTypes} from './item.types';
import apiInstance from '../../axios';

const getListUrl = 'api/v1/items-list?page=1&limit=100&days=$1000';
const addItemUrl = 'api/v1/add-item';

export function* getItemList() {
  try {
    //const { data } = yield call(apiInstance.get, getListUrl);
    const {data} = yield call(apiInstance.get, getListUrl);
    yield put(getItemListSucess(data.data.data));
  } catch (error) {
    console.log(error);
  }
}

export function* addItem({
  payload: {
    item_title,
    item_keywords,
    item_desc,
    item_category,
    item_image,
    image_id,
  },
}) {
  // console.log(
  //   'hello :' + item_title,
  //   'hhi :' + item_keywords,
  //   'image value: ' + image,
  // );
  // noratad658@tenaze.com,   Tested2022
  try {
    const {data} = yield call(apiInstance.post, addItemUrl, {
      item_title,
      item_keywords,
      item_desc,
      item_category,
      item_image,
      image_id,
    });
    console.log(data);
    yield put(addItemSucess(data));
  } catch (error: any) {
    console.log(error.response.data);
    yield put(addItemFail({message: error.response.data}));
  }
}

export function* removeItem({payload: id}) {
  console.log('item _id is loaded', id);
  try {
    const {data} = yield call(apiInstance.delete, `api/v1/delete-item/${id}`);
    console.log(data);
    yield put(removeItemSucess(data.data));
  } catch (error) {
    console.log('get error', error);
  }
}

export function* onGetItemListStart() {
  yield takeLatest(ItemTypes.GET_ITEM_LIST_START, getItemList);
}

export function* onAddItemStart() {
  yield takeLatest(ItemTypes.ADD_ITEM_START, addItem);
}
export function* onRemoveItemStart() {
  yield takeLatest(ItemTypes.REMOVE_ITEM_START, removeItem);
}

export function* itemSagas() {
  yield all([
    call(onGetItemListStart),
    call(onAddItemStart),
    call(onRemoveItemStart),
  ]);
}
