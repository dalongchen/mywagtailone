
<template>
<div>
<div v-for="i in tableData" :key="i.id">
  {{i}}
</div>
</div>
</template>

<script>
  export default {     
      data() {
        const obj = { a:1 , b: 2, c: 3 };
        //const arr = Object.values(obj);
        const arr1 = Object.keys(obj);
        console.log(arr1); 
        return {
          tableData: [{ a:1 , b: 2, c: 3 }, { a:11 , b: 21, c: 31 }],
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
