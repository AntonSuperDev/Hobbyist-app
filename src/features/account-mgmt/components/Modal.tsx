import React, {useState} from 'react';
import {Button, Modal, StyleSheet, View} from 'react-native';
import {WebView} from 'react-native-webview';

const TermsModal = ({showModal, setShowModal}) => {
  return (
    <View style={{flex: 1, justifyContent: 'center', alignItems: 'center'}}>
      <Modal visible={showModal}>
        <View style={styles.modal}>
          <Button
            title="Close"
            onPress={() => setShowModal(false)}
            color="red"
          />
          <View style={styles.modalContainer}>
            <WebView
              style={{flex: 1}}
              source={{
                html: `
           <html>
    
           <head>
               
               <style>
                   
                   body{background-color: #FFFFFF; padding: 40;}
                   
                   p {
                       
                       font-family: Arial, Helvetica, sans-serif;
                       
                       font-size: 40;
                       
                       color:black;
                       
                   }
               
                   </style>
               
           </head><body>
               
               <p><b>Privacy Policy</b></p>
               
               <p>This privacy policy governs your use of the software application Hobbyist.</p>
                     
               <p><strong>Security</strong></p>
               
               <p>We are concerned about safeguarding the confidentiality of your information. We do not use your personal information for purposes other than reaching out to you for issues with the application. Aggregate database information may be mined for insights on the collectibles market but your personal information will never be part of that outside of collectible data.</p>
                   
               <p>If you wish to delete your account email info@hobbyist-app.net.</p>

               <p><strong>Your Consent</strong></p>
               
               <p>By using the Application, you are consenting to our processing of your information as set forth in this Privacy Policy now and as amended by us.</p>

               
               <p><strong>Contact us</strong></p>
               
               <p>If you have any questions regarding privacy while using the Application, please contact info@hobbyist-app.net.</p>
               
           </body>
           
       </html>
            `,
              }}
            />
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  modal: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingTop: 30,
  },
  modalContainer: {
    backgroundColor: 'white',
    width: '90%',
    height: '90%',
  },
  ActivityIndicatorStyle: {
    flex: 1,
    justifyContent: 'center',
  },
});

export default TermsModal;
