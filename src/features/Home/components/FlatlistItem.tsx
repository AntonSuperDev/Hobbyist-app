/* eslint-disable react-hooks/exhaustive-deps */
import {
  ItemDescription,
  ItemDescriptionText,
  ItemDetailImage,
  ItemImage,
  ItemList,
  ItemNavigate,
  ItemTitle,
  ItemTitleText,
  OnTouch,
} from './flatlistitem.styles';
import React, {FC} from 'react';

import ArrowRight from '../../../../assets/chevron-right.png';
import {Image} from 'react-native';
import {Loading} from '../../auth/components/login.styles';
import SampleImage from '../../../../assets/sample-image.png';
import {SectionDivider} from '../../account-mgmt/components/account.mgmt.styles';
import {TouchableOpacity} from 'react-native';

interface Props {
  navigation: any;
  item: {
    _id: string;
    item_title: string;
    item_image: string;
    item_keywords: string;
    item_desc: string;
    average: number;
  };
  idx: number;
  role: any;
}

const FlatlistItem: FC<Props> = ({navigation, item, role}) => {
  const [avg, setAvg] = React.useState('');
  React.useEffect(() => {
    if (item.average === null || undefined || '') {
      setAvg('0.00');
    } else {
      setAvg(item.average);
    }
  }, []);

  return (
    <>
      <ItemList>
        <ItemTitle>
          <ItemImage
            source={
              item.item_image ? {uri: item.item_image} : JSON.parse(SampleImage)
            }
          />
          <ItemTitleText>{item.item_title}</ItemTitleText>
        </ItemTitle>
        {role === 'tier-I' ? (
          <ItemDescription>
            {/* <OnTouch onPress={() => navigation.navigate('AccountMgmt')}> */}
            <TouchableOpacity
              onPress={() => navigation.navigate('AccountMgmt')}>
              <ItemDescriptionText>Upgrade</ItemDescriptionText>
            </TouchableOpacity>
            {/* </OnTouch> */}
          </ItemDescription>
        ) : (
          <ItemDescription>
            {avg !== '' || null ? (
              <ItemDescriptionText>${avg.toFixed(2)}</ItemDescriptionText>
            ) : (
              <ItemDescriptionText>
                <Loading size={14} color="#000" />
              </ItemDescriptionText>
            )}
          </ItemDescription>
        )}
        <ItemNavigate>
          <OnTouch
            onPress={() => navigation.navigate('JobDetail', {item: item})}>
            <ItemDetailImage source={JSON.parse(ArrowRight)} />
          </OnTouch>
        </ItemNavigate>
      </ItemList>
    </>
  );
};

export default FlatlistItem;
