import {
  DeleteIcon,
  ItemDescription,
  ItemNavigate,
  ItemTitle,
  RemoveItemNavigate,
} from './flatlistItemremove.styles';
import {
  ItemDescriptionText,
  ItemDetailImage,
  ItemImage,
  ItemList,
  ItemTitleText,
  OnTouch,
} from '../../Home/components/flatlistitem.styles';
import React, {FC} from 'react';

import ArrowRight from '../../../../assets/chevron-right.png';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Delete from '../../../../assets/delete.png';
import SampleImage from '../../../../assets/sample-image.png';
import {getItemListStart} from '../../../services/items/item.action';
import {useDispatch} from 'react-redux';

type Props = {
  navigation: any;
  deleteItem: any;
  item: {
    _id: string;
    item_title: string;
    item_image: string;
    item_keywords: string;
    item_desc: string;
  };
};
const FlatlistItemRemove: FC<Props> = ({navigation, item, deleteItem,}) => {
  return (
    <ItemList>
      <ItemTitle>
        <ItemImage
          source={
            item.item_image ? {uri: item.item_image} : JSON.parse(SampleImage)
          }
        />
        <ItemTitleText>{item.item_title}</ItemTitleText>
      </ItemTitle>
      <ItemDescription>
        <ItemDescriptionText>{item.item_desc}</ItemDescriptionText>
      </ItemDescription>
      <RemoveItemNavigate>
        <OnTouch onPress={() => deleteItem(item._id)}>
          <DeleteIcon source={JSON.parse(Delete)} />
        </OnTouch>
      </RemoveItemNavigate>
      <ItemNavigate>
        <OnTouch onPress={() => navigation.navigate('JobDetail', {item: item})}>
          <ItemDetailImage source={JSON.parse(ArrowRight)} />
        </OnTouch>
      </ItemNavigate>
    </ItemList>
  );
};

export default FlatlistItemRemove;
