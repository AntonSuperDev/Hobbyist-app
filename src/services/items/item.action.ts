import {ItemTypes} from './item.types';

// Get all item list
export const getItemListStart = () => ({
  type: ItemTypes.GET_ITEM_LIST_START,
});

export const getItemListSucess = (itemList: any) => ({
  type: ItemTypes.GET_ITEM_LIST_SUCCESS,
  payload: itemList,
});

export const getItemListFail = (error: any) => ({
  type: ItemTypes.GET_ITEM_LIST_FAIL,
  payload: error,
});

// Add New Item to List
export const addItemStart = (data: any) => ({
  type: ItemTypes.ADD_ITEM_START,
  payload: data,
});

export const addItemSucess = (item: any) => ({
  type: ItemTypes.ADD_ITEM_SUCCESS,
  payload: item,
});

export const addItemFail = (error: any) => ({
  type: ItemTypes.ADD_ITEM_FAIL,
  payload: error,
});

// Remove Item
export const removeItemStart = (id: string) => ({
  type: ItemTypes.REMOVE_ITEM_START,
  payload: id,
});

export const removeItemSucess = (item: any) => ({
  type: ItemTypes.REMOVE_ITEM_SUCCESS,
  payload: item,
});

export const removeItemFail = (error: any) => ({
  type: ItemTypes.REMOVE_ITEM_FAIL,
  payload: error,
});
