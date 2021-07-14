
<template>{{t}}
</template>

<script>
  export default {
      
      data() {
        const obj = { a:1 , b: 2, c: 3 };
//const arr = Object.values(obj);
const arr1 = Object.keys(obj);
console.log(arr1); 
        return {
          tableData: [['2016-05-02','王小虎']],
          t: [
            {prop: "0", propName: "编号"},
            {prop: "1", propName: "名称"},
          ],
        }
      }
    }
</script>

<style>

</style>
