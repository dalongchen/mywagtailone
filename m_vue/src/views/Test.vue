<template>
<TestSon/>
</template>
<script lang="ts"> 
import { defineComponent } from "vue";
// import { defineComponent, ref } from "vue";
import TestSon from "@/components/TestSon.vue";

export default defineComponent({
  name: "Test",
  components: {
    TestSon
  },
//   setup() {
//     const age02 = ref("32");
//     const name02 = ref("xx");

//     return {
//       age02,
//       name02,
//     };
//   },
//   data() {
//     return {
//       msgFormSon: "this is msg"
//     };
//   },
//   methods: {
//     getMsgFormSon(data: string) {
//       this.msgFormSon = data;
//       console.log("parent" + this.msgFormSon);
//     }
//   }
});
</script>